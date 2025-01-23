import os
import tensorflow as tf

def configure_tensorflow():
    # Configurações de ambiente
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # Configurar threads antes de qualquer outra operação
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Configurar cache de algoritmos cuDNN
    tf.config.optimizer.set_jit(False)  # Desativa XLA JIT
    
    # Configurar GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3072)]
                )
        except RuntimeError as e:
            print(f"Erro ao configurar GPU: {e}")