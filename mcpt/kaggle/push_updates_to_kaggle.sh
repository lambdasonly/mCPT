#!/bin/sh

echo Writing custom_loss.py to custom_classes.py
cat ../contrastlearning/custom_loss.py > custom_classes.py
echo "print('Loaded WeightedCosineSimilarityLoss...')" >> custom_classes.py
echo -e \\n\\n\\n >> custom_classes.py

echo Appending sampler.py to custom_classes.py
cat ../contrastlearning/sampler.py >> custom_classes.py
echo "print('Loaded ContrastSampler...')" >> custom_classes.py
echo -e \\n\\n\\n >> custom_classes.py

echo Appending trainer.py to custom_classes.py
cat ../contrastlearning/trainer.py | sed '/^from \./d' >> custom_classes.py
echo "print('Loaded Trainer...')" >> custom_classes.py
echo -e \\n\\n\\n >> custom_classes.py

echo Appending trainerA.py to custom_classes.py
cat ../contrastlearning/trainerA.py | sed '/^from \./d' >> custom_classes.py
echo "print('Loaded TrainerA...')" >> custom_classes.py
echo -e \\n\\n\\n >> custom_classes.py

echo Appending trainerB.py to custom_classes.py
cat ../contrastlearning/trainerB.py | sed '/^from \./d' >> custom_classes.py
echo "print('Loaded TrainerB...')" >> custom_classes.py
echo -e \\n\\n\\n >> custom_classes.py

echo Appending datamanager.py to custom_classes.py
cat ../contrastlearning/datamanager.py | sed '/^from \./d' >> custom_classes.py
echo "print('Loaded DataManager...')" >> custom_classes.py

if [[ $1 != nopush ]]; then
  echo Pushing custom_classes.py to kaggle...
  kaggle kernels push
fi
echo DONE
