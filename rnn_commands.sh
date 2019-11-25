sudo apt update
sudo apt install python3-pip
pip3 install numpy
# For installing default tensorflow, but this will install 1.14
pip3 install tensorflow
# For tensorflow 2.0 
# This is only the CPU version, for GPU version install the corresponding tensorflow
# Please note it might be useful to work with virtual environments in this case.
sudo python3 -m pip install --upgrade pip && sudo python3 -m pip install --upgrade -v tensorflow==2.0.0rc0

wget https://raw.githubusercontent.com/nithinv13/NLP-with-Python/master/data/small_vocab_en
wget https://raw.githubusercontent.com/nithinv13/NLP-with-Python/master/data/small_vocab_fr


from operator import itemgetter
from pympler import tracker
mem = tracker.SummaryTracker()
print(sorted(mem.create_summary(), reverse=True, key=itemgetter(2))[:10])
memory = pd.DataFrame(mem.create_summary(), columns=['object', 'number_of_objects', 'memory'])
memory['mem_per_object'] = memory['memory'] / memory['number_of_objects']
print(memory.sort_values('memory', ascending=False).head(10))
print(memory.sort_values('mem_per_object', ascending=False).head(10))


python3 rnn_encoder_decoder.py --epochs=1000 --batch_size=8192 --validation_split=0.2 --learning_rate=0.01 --monitor=val_accuracy --patience=5


