import argparse
import getpass


def add_train_arguments(parser: argparse.ArgumentParser):
    _add_common_arguments(parser)
    _add_train_data_args(parser)
    _add_model_args(parser)
    _add_train_args(parser)
    _add_telegram_token(parser)


def add_eval_arguments(parser: argparse.ArgumentParser):
    _add_common_arguments(parser)
    parser.add_argument('--saved_model_path', type=str, help='Path to the trained model')
    parser.add_argument('--data_name', type=str, default='CafeFNewsRecommend', help='Name of the eval dataset')
    parser.add_argument('--eval_behaviors_path', default='data/valid_behaviors.tsv', type=str)
    parser.add_argument('--fast_eval', action='store_true', help='Is there a fast evaluation for the eval dataset?')
    parser.add_argument('--eval_batch_size', type=int, help='How many samples per batch to load in the test phase')
    parser.add_argument('--dataloader_num_workers', type=int, help='How many subprocesses to use for data loading')
    parser.add_argument('--eval_path', type=str, default='eval',
                        help='Path to the directory where evaluation information is stored')


def _add_common_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--pretrained_tokenizer', type=str, help='Path to the pre-trained tokenizer')
    parser.add_argument('--user2id_path', type=str, help='Path to the user dictionary')
    parser.add_argument('--category2id_path', type=str, help='Path to the category dictionary')
    parser.add_argument('--category_embed_path', type=str, default=None,
                        help='Path to the pre-trained category embedding')
    parser.add_argument('--max_title_length', type=int, help='The maximum length of a title encodes')
    parser.add_argument('--max_sapo_length', type=int, help='The maximum length of a sapo encodes')
    parser.add_argument('--his_length', type=int, help='Max number of user click history')
    parser.add_argument('--seed', type=int, help='Seed value')
    parser.add_argument('--save_eval_result', action='store_true', help='Whether or not to save the evaluation result')
    parser.add_argument('--metrics', type=str, nargs='+', help='List of metrics used for evaluation')


def _add_train_data_args(parser: argparse.ArgumentParser):
    parser.add_argument('--data_name', type=str, default=None, help='Name of the train dataset')
    parser.add_argument('--train_behaviors_path', type=str,
                        help='Path to the behaviors.tsv file for the training phase')
    parser.add_argument('--train_news_path', type=str, help='Path to the news.tsv file for the training phase')
    parser.add_argument('--eval_behaviors_path', type=str,
                        help='Path to the behaviors.tsv file for the evaluation phase')
    parser.add_argument('--eval_news_path', type=str, help='Path to the news.tsv file for the evaluation phase')
    parser.add_argument('--fast_eval', action='store_true', help='Is there a fast evaluation for the eval dataset?')


def _add_model_args(parser: argparse.ArgumentParser):
    parser.add_argument('--pretrained_embedding', type=str, help='Path to the pre-trained Roberta model')
    parser.add_argument('--freeze_transformer', action='store_true', help='Whether to freeze Roberta weight or not')
    parser.add_argument('--apply_reduce_dim', action='store_true',
                        help="Whether to reduce the dimension of Roberta's embedding or not")
    parser.add_argument('--use_cls_embed', action='store_true',
                        help='Whether to use the embedding of [CLS] token as a sequence embeddings or not')
    parser.add_argument('--use_sapo', action='store_true', help='Whether to use sapo embedding or not')
    parser.add_argument('--use_category', action='store_true', help='Whether to use category embedding or not')
    parser.add_argument('--query_dim', type=int, help='The size of query vector in the additive attention network')
    parser.add_argument('--dropout', type=float, help='Dropout value')
    parser.add_argument('--category_embed_dim', type=int, help='The size of each category embedding vector')
    parser.add_argument('--num_cnn_filters', type=int, help='The number of output filters in the convolution')
    parser.add_argument('--window_size', type=int, help='The length of the 1D convolution window')
    parser.add_argument('--word_embed_dim', type=int, help='Size of each word embedding vector if apply_reduce_dim')
    parser.add_argument('--num_gru_layers', type=int, help='Number of recurrent layers in GRU network')
    parser.add_argument('--gru_dropout', type=float, help='Dropout value in GRU network')
    parser.add_argument('--long_term_dropout', type=float, help='Dropout value in Long-Term User Representations')
    parser.add_argument('--combine_type', type=str, choices=['ini', 'con'],
                        help='Method to combine the long-term and short-term user representations')


def _add_train_args(parser: argparse.ArgumentParser):
    parser.add_argument('--train_path', type=str, default='train',
                        help='Path to the directory where training information is stored')
    parser.add_argument('--tensorboard_path', type=str, default='runs',
                        help='Path of the directory where to save the log files to be parsed by TensorBoard')
    parser.add_argument('--npratio', type=int, default=4, help='Number of negative samples per positive sample')
    parser.add_argument('--train_batch_size', type=int, help='How many samples per batch to load in the training phase')
    parser.add_argument('--eval_batch_size', type=int,
                        help='How many samples per batch to load in the validation phase')
    parser.add_argument('--dataloader_drop_last', action='store_true',
                        help='Whether to drop the last incomplete batch (if the length of the dataset is not divisible '
                             'by the batch size) or not')
    parser.add_argument('--dataloader_num_workers', type=int, help='How many subprocesses to use for data loading')
    parser.add_argument('--dataloader_pin_memory', action='store_true', help='If True, the data loader will copy '
                                                                             'Tensors into device/CUDA pinned memory '
                                                                             'before returning them.')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='If set to a positive number, the total number of training steps to perform '
                             '(Overrides num_train_epochs)')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether to use fp16 16-bit (mixed) precision training instead of 32-bit training')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                        help='Number of updates steps to accumulate the gradients for, before performing a '
                             'backward/update pass')
    parser.add_argument('--num_train_epochs', type=int, help='Total number of training epochs to perform')
    parser.add_argument('--learning_rate', type=float, help='Learning rate of the optimization algorithm')
    parser.add_argument('--warmup_ratio', type=float,
                        help='Ratio of total training steps used for a linear warmup from 0 to learning_rate')
    parser.add_argument('--warmup_steps', type=int, default=None,
                        help='Number of steps used for a linear warmup from 0 to learning_rate (override warmup_ratio)')
    parser.add_argument('--max_grad_norm', type=float, help='Max norms of the gradients')
    parser.add_argument('--weight_decay', type=float, help='Decoupled weight decay to apply')
    parser.add_argument('--logging_steps', type=int, help='Number of update steps between two logs')
    parser.add_argument('--evaluation_info', type=str, nargs='+', choices=['loss', 'metrics'],
                        help='Evaluation information to log')
    parser.add_argument('--eval_steps', type=int, help='Number of update steps between two evaluations')
    parser.add_argument('--use_telegram_notify', action='store_true',
                        help='Whether to announce the results via telegram or not')


def _add_telegram_token(parser: argparse.ArgumentParser):
    parser.add_argument('--telegram_token', action=Password, nargs='?', dest='Telegram token: ')
    parser.add_argument('--telegram_chat_id', action=Password, nargs='?', dest='Telegram chat ID: ')


class Password(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values is None:
            values = getpass.getpass(self.dest)
        setattr(namespace, self.dest, values)
