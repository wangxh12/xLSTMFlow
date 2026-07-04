from data_provider.data_loader import MinnesotaSegLoader, MinnesotaPredLoader,PredNavAltSegLoader, \
    ThorNavAltPredLoader
from torch.utils.data import DataLoader
from data_provider.alfa import ALFAReconSegLoader, ALFAPredSegLoader


data_dict = {
    'Minnesota': MinnesotaSegLoader,
    'MinnesotaPred': MinnesotaPredLoader,
    "PredNavAlt": PredNavAltSegLoader,
    'ThorNavAltPred': ThorNavAltPredLoader,
    'AlfaPred': ALFAPredSegLoader,
    "AlfaRecon": ALFAReconSegLoader,
    
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    batch_size = args.batch_size
    freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            args = args,
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    
    elif args.task_name == 'anomaly_detection_pred':
        data_set = Data(
            args=args,
            root_path=args.root_path,
            win_size=args.seq_len,
            pred_len=args.pred_len,
            step=1,
            flag=flag,
        )

        print(flag, len(data_set))

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

