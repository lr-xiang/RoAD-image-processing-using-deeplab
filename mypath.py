class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            print("dataset is sbd")
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'arab':
        	return '/home/lietang/Documents/RoAD/pytorch-deeplab-xception/dataloaders/datasets/arab/'
        elif dataset == 'arab3':
        	return '/home/lietang/Documents/RoAD/pytorch-deeplab-xception/dataloaders/datasets/arab3/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
