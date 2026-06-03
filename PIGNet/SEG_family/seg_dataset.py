from GCN.PIGNet.SEG_family.pascal import VOCSegmentation
from GCN.PIGNet.SEG_family.cityscapes import Cityscapes

def get_dataset(config):

    config.train = True if config.mode == "train" else False

    if config.dataset == 'pascal':

        if config.mode == "train":

            print("train dataset")
            dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                      train=config.train, crop_size=config.crop_size)
            valid_dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                            train=not (config.train), crop_size=config.crop_size)
        else:

            if config.infer_params.process_type != None:
                print(config.infer_params.process_type)
                dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                                train=config.train, crop_size=config.crop_size,
                                                process=config.infer_params.process_type, process_value=config.factor,
                                                overlap_percentage=config.factor,
                                                pattern_repeat_count=config.factor,
                                                MI = config.MI)
            else:
                dataset = VOCSegmentation('/home/hail/pan/GCN/PIGNet/data/VOCdevkit',
                                                train=config.train, crop_size=config.crop_size,
                                                process=None, process_value=config.factor,
                                                overlap_percentage=config.factor,
                                                pattern_repeat_count=config.factor,
                                                MI=config.MI)

    elif config.dataset == 'cityscape':

        if config.train:
            print("train dataset cityscape")

            dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                 train=config.train, crop_size=config.crop_size)

            valid_dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                 train=not (config.train), crop_size=config.crop_size)

        else: # val
            if config.infer_params.process_type != None:
                print(config.infer_params.process_type)
                dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                          train=config.train, crop_size=config.crop_size,
                                          process=config.infer_params.process_type, process_value=config.factor,
                                          overlap_percentage=config.factor,
                                          pattern_repeat_count=config.factor,
                                          MI = config.MI)
            else:
                dataset = Cityscapes('/home/hail/pan/GCN/PIGNet/data/cityscape',
                                          train=config.train, crop_size=config.crop_size,
                                          process=None, process_value=config.factor,
                                          overlap_percentage=config.factor,
                                          pattern_repeat_count=config.factor)

    else:
        raise ValueError('Unknown dataset: {}'.format(config.dataset))
    
    if config.train:
        return dataset, valid_dataset
    else:
        return dataset