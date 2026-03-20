#!/bin/bash
# One-command retrain pipeline:
# 1. Pull false positive clips from Pi
# 2. Add to negative training set
# 3. Repackage training_data.zip
# 4. Upload to Google Drive (replacing old zip)
# 5. Print Colab link to run

set -e
cd "$(dirname "$0")"
source venv/bin/activate

echo "=== Step 1: Pull detection clips from Pi ==="
mkdir -p false_positives/detection_clips
scp -i ~/.ssh/minazap_pi "del@192.168.10.10:~/minazap/detection_clips/*.wav" ./false_positives/detection_clips/ 2>/dev/null || true
CLIP_COUNT=$(ls false_positives/detection_clips/*.wav 2>/dev/null | wc -l | tr -d ' ')
echo "  Pulled $CLIP_COUNT clips"

echo ""
echo "=== Step 2: Add to negative training set ==="
cp false_positives/detection_clips/*.wav training_data/negative/ 2>/dev/null || true
NEG_COUNT=$(find training_data/negative -name "*.wav" | wc -l | tr -d ' ')
echo "  Total negatives: $NEG_COUNT"

echo ""
echo "=== Step 3: Package training data ==="
python package_for_colab.py

echo ""
echo "=== Step 4: Upload to Google Drive ==="
rclone copy training_data.zip gdrive: --progress
echo "  Upload complete"

echo ""
echo "=== Step 5: Clear Pi detection clips ==="
ssh -i ~/.ssh/minazap_pi del@192.168.10.10 "rm -f ~/minazap/detection_clips/*.wav" 2>/dev/null || true
rm -f false_positives/detection_clips/*.wav
echo "  Clips cleared"

echo ""
echo "=== Done! ==="
echo "Open Colab and run all cells:"
echo "  https://colab.research.google.com/github/deloschang/minazapper/blob/main/train_colab.ipynb"
echo ""
echo "After training, deploy to Pi:"
echo "  scp -i ~/.ssh/minazap_pi mina_classifier.tflite del@192.168.10.10:~/minazap/"
echo "  ssh -i ~/.ssh/minazap_pi del@192.168.10.10 'pkill -f bark_detector; source ~/minazap-env/bin/activate && DISPLAY=:0 nohup python -u ~/minazap/bark_detector.py --threshold 0.85 --silence 30 > ~/minazap/detector.log 2>&1 &'"
