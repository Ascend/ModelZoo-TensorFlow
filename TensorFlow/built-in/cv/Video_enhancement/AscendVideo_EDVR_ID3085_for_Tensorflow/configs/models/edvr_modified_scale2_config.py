cfg = dict(
    data=dict(
        data_dir='/data2/vsr_datasets/reds',
        train=dict(
            batch_size=16,
            input_size=[64, 64],
            augmentation=dict(
                apply=True,
                interval_list=[1,2,3],
                options="""
                        RandomCrop:
                            input_dim: 4
                        RandomTemporalReverse:
                            input_dim: 4
                        RandomFlipLeftRight:
                            input_dim: 4
                        RandomFlipUpDown:
                            input_dim: 4
                        """,
            ),
        ),
        inference=dict(
            input_size=[288, 352]
        )
    ),
    edvr=dict(
        with_tsa=False,
        mid_channels=64,
        use_dcn=False,
        num_groups=1,
        num_deform_groups=1,
        num_blocks_extraction=5,
        num_blocks_reconstruction=10,
        upsampling='bilinear',
        align_corners=True
    ),
    model=dict(
        content_loss_reduction='mean',
        content_loss_type='l1',
        factor_for_adapt_input=4,
        name='EDVRVariant',
        num_net_input_frames=3,
        num_net_output_frames=1,
        scale=2,
        scope='G'
    ),
    loss=dict(
        content=dict(
            loss_type='L1Loss',
            loss_reduction='mean'
        ),
    ),
    train=dict(
        print_interval=100,
        output_dir='output/edvr',
        generator=dict(
            lr_schedule=dict(
                total_steps=[10000]
            )
        )
    ),
    log_file='edvr_train.log',
)