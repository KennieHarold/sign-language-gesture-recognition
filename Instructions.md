# Extracting frames form training videos 
```shell
python "video-to-frame.py" train train_frames
```

# Extracting frames form test videos
```shell
python "video-to-frame.py" test test_frames
```

# Retrain the Inception v3 model
```shell
python retrain.py --bottleneck_dir=bottlenecks --summaries_dir=training_summaries/long --output_graph=retrained_graph.pb --output_labels=retrained_labels.txt --image_dir=train_frames
```

# Intermediate representation of videos for train frames
```shell
python predict_spatial.py retrained_graph.pb train_frames --batch=100
```

# Intermediate representation of videos for train frames
```shell
python predict_spatial.py retrained_graph.pb test_frames --batch=100
```

# Train the RNN
```shell
python rnn_train.py predicted-frames-final_result-train.pkl non_pool.model
```

# Test the RNN Model
```shell
python3 rnn_eval.py predicted-frames-final_result-test.pkl non_pool.model
```