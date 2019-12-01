from rnn_encoder_decoder import EncoderDecoder
import contextlib
import time

if __name__ == '__main__':
    '''
    driver program for training
    '''
    lr = [1e-5, 0.1] #specify start and end
    bs = [512, 16384] #specify start and end
    learning_rate = lr[0]
    batch_size = bs[0]
    trainer = EncoderDecoder(epochs=500, batch_size=batch_size, learning_rate=learning_rate, validation_split=0.2, monitor='val_acc', min_delta=0.001, patience=50)
    trainer.print_device()
    trainer.loadData()
    trainer.seeSampleData()
    trainer.prepareData()
    while learning_rate <= lr[1]:
        while batch_size <= bs[1]:
            with open('logs/GPU_Utils.txt', 'a') as f:
                with contextlib.redirect_stdout(f):
                    print('Batch Size: {} Learning Rate: {}'.format(batch_size, learning_rate))
            #trainer = EncoderDecoder(epochs=2, batch_size=batch_size, learning_rate=learning_rate, validation_split=0.2, monitor='val_acc', min_delta=0.001, patience=50)
            trainer.setBatchSize(batch_size)
            trainer.setLearningRate(learning_rate)
            trainer.train_model()
            #time.sleep(300) #sleep for 5 min => To know difference in graph
            batch_size = batch_size * 2
        learning_rate = learning_rate * 10

