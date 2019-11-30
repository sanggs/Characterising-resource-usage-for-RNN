from rnn_encoder_decoder import EncoderDecoder

if __name__ == '__main__':
    '''
    driver for the training to go on
    '''
    trainer = EncoderDecoder(epochs=2, batch_size=8192, learning_rate=0.01, validation_split=0.2, monitor='val_accuracy', min_delta=1.0, patience=5)
    trainer.train_model()

