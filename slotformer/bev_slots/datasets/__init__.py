from .bev import build_obj3d_dataset, build_obj3d_slots_dataset
from .clevrer import build_clevrer_dataset, build_clevrer_slots_dataset
from .physion import build_physion_dataset, build_physion_slots_dataset, \
    build_physion_slots_label_dataset
from .phyre import build_phyre_dataset, build_phyre_slots_dataset, \
    build_phyre_rollout_slots_dataset
from .bevdataset import CarlaVoiceDataset

def create_carla_dataset(
    root, towns, weathers=None, waypoints_seq_len=5, batch_size=None, **kwargs
):
    ds = CarlaVoiceDataset(
                dataset_root="/home/matias/slotformer/data/bev",
                towns=None,
                weathers=None,
                scale=None,
                enable_start_frame_augment=True,
                token_max_length=2, # THESIS: this is 40 in the original
                enable_notice=True,
        )
    return ds

def build_dataset(params, val_only=False):
    dst = params.dataset
    dataset_train = create_carla_dataset(
        root="/home/matias/slotformer/data/bev",
        towns=None,
        weathers=None,
        batch_size=24,
        with_lidar=False,
        with_seg=False,
        with_depth=False,
        multi_view=False,
        augment_prob=False,
        temporal_frames=False,
    )
    dataset_eval = create_carla_dataset(
        root="/home/matias/slotformer/data/bev",
        towns=[1],
        weathers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19],
        batch_size=24,
        with_lidar=False,
        with_seg=False,
        with_depth=False,
        multi_view=False,
        augment_prob=False,
        temporal_frames=False,
    )
    return dataset_train, dataset_eval
