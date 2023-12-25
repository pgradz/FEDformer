import os
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import FEDformer, Autoformer, Informer, Transformer
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'FEDformer': FEDformer,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.classifier:
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cls_y, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cls_y = cls_y.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0

                outputs = outputs.to(torch.float64)
                cls_y  = cls_y.to(torch.float64)

                if not self.args.classifier:
                    # close price is at position 1
                    last_close_price = batch_y[:,  -self.args.pred_len-1, 1].to(self.device)
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    close_price_pred = outputs[:, :, 1]
                    positive_diff = abs(close_price_pred.max(dim=1)[0] - last_close_price)
                    negative_diff = abs(close_price_pred.min(dim=1)[0] - last_close_price)
                    direction = (positive_diff > negative_diff).float()
                    direction = direction.unsqueeze(1)
                    true_cls = cls_y.detach().cpu()
                    total += true_cls.size(0)
                    correct += (direction == true_cls).sum().item()

                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()
                    loss = criterion(pred, true)

                else:
                    # it can happen that prediction is Nan
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    nan_mask = torch.isnan(outputs)
                    num_nans = torch.sum(nan_mask).item()
                    if num_nans > 0:
                        print('Warning: {} NaNs in validation'.format(num_nans))
                        outputs[nan_mask] = 0.5

                    pred = outputs.detach().cpu()
                    true = cls_y.detach().cpu()
                    predicted_class = (pred > 0.5).float()
                    total += true.size(0)
                    correct += (predicted_class == true).sum().item()
                    loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        accuracy = (100 * correct / total)
        self.model.train()
        return total_loss, accuracy

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cls_y, _) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cls_y = cls_y.float().to(self.device)

                # decoder input
                # creates empty tensor of shape (batch_size, pred_len, n_features)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # creates tensor with label_len acutal values and pred_len zeros
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    # check if output returns NaN
                    nan_mask = torch.isnan(outputs)
                    num_nans = torch.sum(nan_mask).item()
                    if num_nans > 0:
                        print('Warning: {} NaNs in training'.format(num_nans))
                        outputs[nan_mask] = 0.5

                    f_dim = -1 if self.args.features == 'MS' else 0
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    if self.args.classifier:
                        loss = criterion(outputs, cls_y)
                    else:
                        loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Val accuracy: {5:.7f} Test accuracy: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, val_accuracy, test_accuracy))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def test(self, setting, test=0):
        if self.args.classifier:
            self.test_classifier(setting, test=test)
        else:
            self.test_triple_barrier(setting, test=test)

    def test_classifier(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        time_indices = []
        correct = 0
        total = 0
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cls_y, time_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cls_y = cls_y.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.to(torch.float64)
                cls_y  = cls_y.to(torch.float64)
                time_index = time_index.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                batch_y = cls_y.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                predicted_class = (pred > 0.5)
                total += true.shape[0]
                correct += (predicted_class == true).sum().item()
                
                pred = np.squeeze(pred.tolist()).tolist()
                true = np.squeeze(true.tolist()).tolist()
                time_index = np.squeeze(time_index.tolist()).tolist()
                preds.append(pred)
                trues.append(true)
                time_indices.append(time_index)
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = [item for sublist in preds for item in sublist]
        trues = [item for sublist in trues for item in sublist]
        time_indices = [item for sublist in time_indices for item in sublist]
        time_stamps = test_data.date_data[time_indices]
        time_stamps = time_stamps.reshape(-1)
        df = pd.DataFrame({'preds': preds, 'trues': trues, 'time_stamps': time_stamps})

        lower_bounds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        upper_bounds = [1 - lower for lower in lower_bounds]
        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            self.accuracy_by_threshold(df, lower_bound, upper_bound)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        accuracy = (100 * correct / total)
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        print('accuracy:{}'.format(accuracy))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        df.to_csv(folder_path + 'predictions_Exp_{}_{}.csv'.format(self.args.ii, self.args.test_end_str))

        return
    

    def test_triple_barrier(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        max_pred_price_list = []
        min_pred_price_list = []
        last_close_price_list = []
        barrier_list = []
        preds = []
        trues = []
        time_indices = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, cls_y, time_index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                cls_y = cls_y.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0

                last_close_price = batch_y[:,  -self.args.pred_len-1, :]
                inverse_last_close_price = test_data.inverse_transform(last_close_price)[:,1]
                last_close_price_list.append(inverse_last_close_price.tolist())

                inverse_outputs = self.get_real_predictions(outputs, test_data)
                close_price_pred = inverse_outputs[:, :, 1]
                max_pred_price = close_price_pred.max(dim=1)[0].detach().cpu().numpy().tolist()
                max_pred_price_list.append(max_pred_price)
                min_pred_price = close_price_pred.min(dim=1)[0].detach().cpu().numpy().tolist()
                min_pred_price_list.append(min_pred_price)
                barrier = self.get_barrier_breached(inverse_last_close_price, close_price_pred)
                barrier_list.append(barrier)

                cls_y  = cls_y.to(torch.float64)
                time_index = time_index.detach().cpu().numpy()
                batch_y = cls_y.detach().cpu().numpy()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                true = np.squeeze(true.tolist()).tolist()
                time_index = np.squeeze(time_index.tolist()).tolist()
                trues.append(true)
                time_indices.append(time_index)

        trues = [item for sublist in trues for item in sublist]
        max_pred_prices = [item for sublist in max_pred_price_list for item in sublist]
        min_pred_prices = [item for sublist in min_pred_price_list for item in sublist]
        last_close_prices = [item for sublist in last_close_price_list for item in sublist]
        barriers = [item for sublist in barrier_list for item in sublist]
        time_indices = [item for sublist in time_indices for item in sublist]
        time_stamps = test_data.date_data[time_indices]
        time_stamps = time_stamps.reshape(-1)
        df = pd.DataFrame({'trues': trues, 'last_close_prices': last_close_prices,
                           'max_pred_prices': max_pred_prices, 
                           'min_pred_prices':min_pred_prices, 'barriers':barriers,
                            'time_stamps': time_stamps})
        
        df['preds'] = df.apply(lambda x: 1 if x['barriers']=='top' else 0 if x['barriers']=='bottom' else 0.5, axis=1)


        lower_bounds = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        upper_bounds = [1 - lower for lower in lower_bounds]
        for lower_bound, upper_bound in zip(lower_bounds, upper_bounds):
            self.accuracy_by_threshold(df, lower_bound, upper_bound)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        # print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        df.to_csv(folder_path + 'predictions_Exp_{}_{}.csv'.format(self.args.ii, self.args.test_end_str))

        return

    

    def get_real_predictions(self, outputs, dataset):
        '''
        Inverse transform the predictions to get the real values
        '''
        tensor_2d = outputs.reshape(-1, outputs.size(2))
        tensor_2d_np = tensor_2d.numpy()
        inversed_np = dataset.inverse_transform(tensor_2d_np)
        inversed_tensor_2d = torch.tensor(inversed_np)
        inversed_tensor_3d = inversed_tensor_2d.view(outputs.shape)

        return inversed_tensor_3d


    def get_barrier_breached(self,inverse_last_close_price, close_price_pred):
        '''
        Find the first barrier breached by the predictions
        '''
        inverse_last_close_price_torch = torch.tensor(inverse_last_close_price)
        inverse_last_close_price_2d = inverse_last_close_price_torch.unsqueeze(1).repeat(1, self.args.pred_len) 
        price_ratio = close_price_pred/inverse_last_close_price_2d
        barrier = self.find_first_barrier(price_ratio, self.args.barrier_threshold)
        return barrier


    def find_first_barrier(self,tensor, threshold):
        # List to store the first deviating element for each row
        first_breach = []

        # Iterate through each row
        for row in tensor:
            # Calculate the absolute deviation from one
            deviation = torch.abs(row - 1)

            # Find the index of the first element with deviation greater than the threshold
            indices = torch.where(deviation > threshold)[0]

            # Check if any element meets the criterion and store the result
            if len(indices) > 0:
                first_index = indices[0].item()
                first_element = row[first_index].item()
                if first_element > 1:
                    first_breach.append('top')
                else:
                    first_breach.append('bottom')
            else:
                first_breach.append('vertical')

        return first_breach


    @staticmethod
    def accuracy_by_threshold(df, lower_bound, upper_bound):
            
        df['y_pred'] = df['preds'].apply(lambda x: 1 if x > 0.5 else 0)
        predictions_above_threshold = df.loc[(df['preds'] < lower_bound) | (df['preds'] > upper_bound)]
        accuracy = (predictions_above_threshold['trues'] == predictions_above_threshold['y_pred']).mean()
        print('Accuracy for predictions outside of [{}, {}]: {:.2f}%'.format(lower_bound, upper_bound, 100 * accuracy))
        return None


    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
