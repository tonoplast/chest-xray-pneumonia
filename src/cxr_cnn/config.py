from pydantic import BaseModel
import yaml

class DataConfig(BaseModel):
    image_size: int = 64
    batch_size: int = 32
    num_workers: int = 0 # TODO: might change this later depending on what happens
    
class TrainingConfig(BaseModel):
    epochs: int = 10
    learning_rate: float = 1e-3
    seed: int = 1337
    
class Config(BaseModel):
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()

def load_config(path: str) -> Config:
    ''' load config from a yaml file '''
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return Config(**raw)