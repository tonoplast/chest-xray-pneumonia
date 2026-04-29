
import logging
import sys

from cxr_cnn.config import load_config
from cxr_cnn.model import PneumoniaCNN, PneumoniaResNetCNN
from cxr_cnn.train import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout),]
)

logger = logging.getLogger(__name__)

def main() -> None:
    ''' training '''
    config = load_config('configs/default.yaml')
    
    logger.info(f'loaded config: {config.model_dump()}')
    
    model_name = config.training.model
    logger.info(f'using model: {model_name}')
    
    if model_name == 'cnn':
        model = PneumoniaCNN(image_size=config.data.image_size)
    elif model_name == 'resnet':
        model = PneumoniaResNetCNN(image_size=config.data.image_size)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    
    train(model, config)
    
if __name__ == '__main__':
    main()
    
    

