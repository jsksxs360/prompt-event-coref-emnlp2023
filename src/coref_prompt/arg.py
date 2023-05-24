import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory where the model checkpoints and predictions will be saved.",
    )
    parser.add_argument("--train_file", default=None, type=str, required=True, help="The input training file.")
    parser.add_argument("--train_file_with_cos", default=None, type=str, required=False, help="Input training file with event similarities.")
    parser.add_argument("--dev_file", default=None, type=str, required=True, help="The input evaluation file.")
    parser.add_argument("--test_file", default=None, type=str, required=True, help="The input testing file.")
    # matched similar triggers and extracted arguments
    parser.add_argument("--train_simi_file", default=None, type=str, required=True, help="The input related info file.")
    parser.add_argument("--dev_simi_file", default=None, type=str, required=True, help="The input related info file.")
    parser.add_argument("--test_simi_file", default=None, type=str, required=True, help="The input related info file.")
    parser.add_argument("--pred_test_simi_file", default=None, type=str, required=True, help="The input related info file.")
    # whether to use sampling to refine the dataset
    parser.add_argument("--sample_strategy", default="corefnm", type=str, required=True, 
        help="The chosen sampling strategy.", choices=['no', 'random', 'corefnm', 'corefenn-1', 'corefenn-2']
    )
    parser.add_argument("--neg_top_k", default="3", type=int)
    parser.add_argument("--neg_threshold", default="0.2", type=float)
    
    parser.add_argument("--model_type", default="roberta", type=str, required=True, choices=['bert', 'roberta'])
    parser.add_argument("--model_checkpoint", default="roberta-large", type=str, required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--cache_dir", default=None, type=str,
        help="Where do you want to store the pre-trained models downloaded from s3."
    )
    parser.add_argument("--max_seq_length", default=512, type=int, required=True)
    parser.add_argument("--prompt_type", type=str, required=True)
    parser.add_argument("--select_arg_strategy", type=str, required=True, choices=['no_filter', 'filter_related_args', 'filter_all'])
    parser.add_argument("--with_mask", action="store_true", help="Whether to input the extra prompt with all triggers masked.")
    parser.add_argument("--matching_style", default="none", type=str, required=True, 
        help="Use tensor matching to help predict masked words.", choices=['no', 'product', 'cosine', 'product_cosine']
    )
    parser.add_argument("--cosine_space_dim", type=int, help="Reduce event embedding dimension.")
    parser.add_argument("--cosine_slices", type=int, help="Cosine matching perspectives.")
    parser.add_argument("--cosine_factor", type=int, help="Tensor factorization.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to predict labels.")
    
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
        help="Proportion of training to perform linear learning rate warmup for, E.g., 0.1 = 10% of training."
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    args = parser.parse_args()
    return args
