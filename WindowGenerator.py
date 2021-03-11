class WindowGenerator: # from TF tutorial
    def __init__(self,
                 input_width,
                 label_width, 
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 label_columns=None):
        # store raw data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # work out the label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # work out the window parameters
        self.input_width = input_width # steps from the past
        self.label_width = label_width # steps into the future -- https://youtu.be/9gF2UySGZAU
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None) # None -> until the end
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.total_window_size = input_width + shift

    def split_window(self, features):
        inputs = features[:, self.input_indices, :]
        labels = features[:, self.labels_slice, :]

        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:,:,self.column_indices[name]] for name in self.label_columns],
                axis=-1)
        
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_indices, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_datasets(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.total_window_size,
                sequence_stride=1,
                shuffle=True,
                batch_size=32,)
        ds = ds.map(self.split_window)

    @property
    def train(self):
        return self.make_datasets(self.train_df)

    @property
    def val(self):
        return self.make_datasets(self.val_df)

    @property
    def test(self):
        return self.make_datasets(self.test_df)
    
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'
        ])