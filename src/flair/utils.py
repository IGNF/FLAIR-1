import yaml
from pytorch_lightning.utilities.rank_zero import rank_zero_only  


def read_config(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)
     

@rank_zero_only
def print_recap(config: dict, dict_train=None, dict_val=None, dict_test=None) -> None:
    """Log content of given config using a tree structure."""

    def walk_config(config: dict, prefix=''):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, dict):
                print(f'{prefix}|- {group_name}:')
                walk_config(group_option, prefix=prefix+'|   ')
            elif isinstance(group_option, list):
                print(f'{prefix}|- {group_name}: {group_option}')
            else:
                print(f'{prefix}|- {group_name}: {group_option}')

    print('Configuration Tree:')
    walk_config(config, '')

    # Log data split information
    print('[---DATA SPLIT---]')
    if config['tasks']['train']:
        for split_name, d in zip(["train", "val"], [dict_train, dict_val]): 
            print(f"- {split_name:25s}: {'':3s}{len(d['IMG'])} samples")
    if config['tasks']['predict']:
        print(f"- {'test':25s}: {'':3s}{len(dict_test['IMG'])} samples")





