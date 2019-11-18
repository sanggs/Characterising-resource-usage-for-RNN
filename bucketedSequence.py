#code reference: https://www.kaggle.com/tyagit3/training-speed-bucketing-variable-len-padding
class BucketedDataIterator():
	def __init__(self, df, num_buckets = 10):
		df = df.sort_values('length').reset_index(drop=True)
		self.size = len(df) / num_buckets
		self.dfs = []
		for bucket in range(num_buckets):
			self.dfs.append(df.loc[bucket*self.size: (bucket+1)*self.size - 1])
		self.num_buckets = num_buckets

		self.cursor = np.array([0] * num_buckets)
		self.shuffle()

		self.epochs = 0

	def shuffle(self):
		for i in range(self.num_buckets):
			self.dfs[i] = self.dfs[i].sample(frac=1).reset_index(drop=True)
			self.cursor[i] = 0

	def next_batch(self, n):
		if np.any(self.cursor+n+1 > self.size):
			self.epochs += 1
			self.shuffle()

		i = np.random.randint(0,self.num_buckets)

		res = self.dfs[i].loc[self.cursor[i]:self.cursor[i]+n-1]
		self.cursor[i] += n

		maxTrainLen = max(res['lengthTrain'])
		maxTestLen = max(res['lengthTest'])
		maxlen = max(maxTrainLen, maxTestLen)
		x_tr = np.zeros([n, maxlen], dtype=np.int32)
		x_te = np.zeros([n, maxlen], dtype=np.int32)
		for i, x_i in enumerate(x_tr):
			x_i[:res['lengthTrain'].values[i]] = res['iTrainData'].values[i]
		for i, x_i in enumerate(x_te):
			x_i[:res['lengthTest'].values[i]] = res['iLabelData'].values[i]
		return x_tr, x_te, maxlen