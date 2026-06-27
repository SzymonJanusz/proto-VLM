import argparse


def read_args():
    parser = argparse.ArgumentParser(description='The options of the MCPNet.')

    parser.add_argument("--index", type=str, default=None, required=True, help="Name of the experiments")
    parser.add_argument("--saved_dir", default=".", type=str)
    parser.add_argument("--log_type", default=["std", "log"], type=str, nargs="+")
    parser.add_argument("--wandb", default=False, action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="MCPNet",
                        help="W&B project name")

    # training hyper parameters
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP parameter. (Don't modify !!)")
    parser.add_argument("--devices", type=int, default=None, required=True, nargs="+")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default=None, required=True, choices=["adam", "sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=int, default=20)
    parser.add_argument("--resume", default=False, action="store_true")

    # model setting
    parser.add_argument("--parameter_path", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default=None, help="Resume parameter path.")
    parser.add_argument("--model", type=str, default=None, required=True, help="File name of the used model.")
    parser.add_argument("--basic_model", type=str, default=None, required=True, help="Class name of the model.")

    # dataset
    parser.add_argument("--dataloader", type=str, default="load_data_train_val_classify")
    parser.add_argument("--dataset_name", type=str, default=None, required=True)
    parser.add_argument("--train_dataset_path", type=str, default=None,
                        help="Override train dataset path (overrides dataset_name lookup)")
    parser.add_argument("--val_dataset_path", type=str, default=None,
                        help="Override val dataset path (overrides dataset_name lookup)")
    parser.add_argument("--mean", type=float, default=[0.485, 0.456, 0.406])
    parser.add_argument("--std", type=float, default=[0.229, 0.224, 0.225])

    ## training set
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--train_num_workers", type=int, default=8)

    ## validation set
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--val_num_workers", type=int, default=8)

    # MCPNet setting
    parser.add_argument('--concept_per_layer', default=[8, 16, 32, 64], type=int, nargs="+")
    parser.add_argument('--concept_cha', default=[32, 32, 32, 32], type=int, nargs="+")
    parser.add_argument("--m_is_concept_num", default=False, action="store_true")

    # Class-aware Concept Distribution (CCD) loss setting
    parser.add_argument("--margin", type=float, default=0.01, help="margin")
    parser.add_argument("--CCD_weight", type=float, default=100.)

    args = parser.parse_args()
    # Save user-provided overrides before model-specific defaults are applied
    user_parameter_path = args.parameter_path
    user_train_path = args.train_dataset_path
    user_val_path = args.val_dataset_path

    if args.basic_model == "resnet50":
        args.parameter_path = "../pretrained/resnet50.pth"
    elif args.basic_model == "resnet18":
        args.parameter_path = "../pretrained/resnet18.pth"
    elif args.basic_model == "resnet34":
        args.parameter_path = "../pretrained/resnet34.pth"
    elif args.basic_model == "resnet50_relu":
        args.parameter_path = "../pretrained/resnet50.pth"
    elif args.basic_model == "resnet152":
        args.parameter_path = "../pretrained/resnet152.pth"
    elif args.basic_model == "convnext_base":
        args.parameter_path = "../pretrained/convnext_base_1k_224_ema.pth"
    elif args.basic_model == "convnext_small":
        args.parameter_path = "../pretrained/convnext_small_1k_224_ema.pth"
    elif args.basic_model == "convnext_tiny":
        args.parameter_path = "../pretrained/convnext_tiny_1k_224_ema.pth"
    elif args.basic_model == "inceptionv3":
        args.parameter_path = "../pretrained/inception_v3.pth"

    # Allow CLI to override hardcoded parameter path
    if user_parameter_path is not None:
        args.parameter_path = user_parameter_path

    if args.dataset_name == "CUB_200_2011":
        args.category = 200
        args.train_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/val"
    elif args.dataset_name == "CUB_200_2011_s":
        args.category = 160
        args.train_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/seen/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/CUB_200_2011/seen/val"
    elif args.dataset_name == "Oxford_Flowers_102":
        args.category = 102
        args.train_dataset_path = "/eva_data_4/bor/datasets/flowers102/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/flowers102/test"
    elif args.dataset_name == "AWA2":
        args.category = 50
        args.train_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/val"
    elif args.dataset_name == "AWA2_s":
        args.category = 40
        args.train_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/seen/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/Animals_with_Attributes2/JPEGImages/seen/val"
    elif args.dataset_name == "Caltech101":
        args.category = 101
        args.train_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/val"
    elif args.dataset_name == "Caltech101_s":
        args.category = 81
        args.train_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/seen/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/101_ObjectCategories/seen/val"
    elif args.dataset_name == "ImageNet-1k-sampled":
        args.category = 1000
        args.train_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/train_sampled"
        args.val_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/val"
    elif args.dataset_name == "ImageNet-1k":
        args.category = 1000
        args.train_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/train"
        args.val_dataset_path = "/eva_data_4/bor/datasets/ImageNet2012/val"
    elif args.dataset_name == "COCO":
        pass  # category inferred from ImageFolder at runtime

    # Allow CLI to override hardcoded dataset paths
    if user_train_path is not None:
        args.train_dataset_path = user_train_path
    if user_val_path is not None:
        args.val_dataset_path = user_val_path

    if args.basic_model != "inceptionv3":
        args.train_random_sized_crop = 224
        args.val_image_size = 224
    else:
        args.train_random_sized_crop = 299
        args.val_image_size = 299

    return args
