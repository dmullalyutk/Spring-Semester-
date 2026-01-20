#Bonus assignment 0: Thought Starter 

#One of the most complex problems in data science is learning models from data streams. 
#Many companies make data accessible inside the organization through APIs.
#A streaming API streams events using push technology by providing a subscription mechanism for receiving events in 
#near real time. 

#Example of a streaming API:
#requires 'numpy' and 'time' to be installed
import requests
url = 'http://ballings.co/firehose.py'
exec(requests.get(url).content)
#hit 'control c' to stop the stream

#Requirements for a learning system:
#(1) There may be periods when data streams in really slowly, and periods when data is pouring in. 
#The system needs to be learning in both situations.
#(2) We want to learn the most accurate model and keep accuracy high at all times.
#(3) We want to keep computational costs low.

#Given everything you have learned in previous classes, design a learning system that aims to satisfy 
#the above requirements. 
#Describe your learning system, and how it satisfies the requirements in 250 words or less. 
#If your system has tradeoffs, add these to your description.

#Note: Full credit can be earned on this bonus assignment even if your system does not meet the requirements, 
#as long as you provide details as to why it does not meet the requirements,


# Answer:
# The proposed system uses an incremental tree-based ensemble with adaptive batch processing
# to learn from streaming data efficiently.
#
# Architecture: Stream data enters a buffer that flushes based on adaptive thresholds. During
# slow periods, small batches (50-100 samples) are processed for rapid model updates. During
# data floods, batch size increases (500-1000 samples) to reduce per-sample computational overhead.
# The core learner is an ensemble of 5-10 Hoeffding Trees (Very Fast Decision Trees), which learn
# incrementally from each batch. A sliding validation window (last 1000 samples) continuously
# monitors model accuracy. When performance drops below a threshold (e.g., 85% of peak accuracy),
# the system triggers model refresh or adds new trees to the ensemble.
#
# How requirements are met:
# (1) Variable flow: The adaptive buffer and dynamic batch sizing ensure continuous learning
# regardless of stream velocity. Buffer absorbs bursts while maintaining steady processing.
# (2) High accuracy: The ensemble approach reduces prediction variance. Performance monitoring
# detects concept drift and triggers timely model updates to maintain accuracy.
# (3) Low computational cost: Hoeffding Trees are computationally efficient for incremental
# learning. Larger batches during floods amortize overhead costs per sample.
#
# Tradeoffs: The primary tradeoff is accuracy vs. speed. During extreme data floods, larger
# batches delay adaptation to concept drift, potentially sacrificing 2-5% accuracy for
# computational feasibility. The ensemble also increases memory usage and computation compared
# to a single model, though this remains manageable with 5-10 trees.