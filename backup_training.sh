#!/bin/bash
# Hourly backup of training data + labels
# Keeps 1 week (168 hours), auto-cleans older ones

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKUP_DIR="$SCRIPT_DIR/backups"
mkdir -p "$BACKUP_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/training_backup_${TIMESTAMP}.tar.gz"

# Compress training data + labels (tar.gz is faster than zip for many small files)
tar czf "$BACKUP_FILE" \
    -C "$SCRIPT_DIR" \
    training_data/mina \
    training_data/negative \
    manual_labels_training.json \
    manual_labels.json \
    2>/dev/null

SIZE=$(du -sh "$BACKUP_FILE" 2>/dev/null | cut -f1)
echo "Backup created: $BACKUP_FILE ($SIZE)"

# Delete backups older than 7 days
find "$BACKUP_DIR" -name "training_backup_*.tar.gz" -mtime +7 -delete 2>/dev/null

echo "Backups: $(ls "$BACKUP_DIR"/training_backup_*.tar.gz 2>/dev/null | wc -l | tr -d ' ') files"
