# Maintenance & Troubleshooting Guide

This document provides detailed procedures for maintaining and troubleshooting OptiMat Alloys.

## Checking for Stale Processes

If Chainlit is not responding or you see port conflicts, stale processes may be running.

### Identify Running Processes

```bash
# Check for running Chainlit processes
ps aux | grep chainlit | grep -v grep

# Example output:
# user     12345  0.5  2.1 1234567 123456 pts/0  Sl+  10:30  0:15 chainlit run run_chat.py
```

### Check Port Usage

```bash
# Check what's using a specific port (e.g., 8000)
lsof -ti:8000

# Example output:
# 12345
```

### Kill Stale Processes

```bash
# Kill process by port (recommended)
kill -9 $(lsof -ti:8000)

# Kill process by PID
kill -9 12345

# Kill all chainlit processes (use with caution)
pkill -9 -f "chainlit run"
```

### Verify Cleanup

```bash
# Confirm port is now free
lsof -ti:8000
# (no output = port is free)

# Confirm no chainlit processes
ps aux | grep chainlit | grep -v grep
# (no output = no processes)
```

## Database Management

### Database Location

```
structures/database.db
```

**Format**: SQLite database with ASE schema

### Check Database Size

```bash
# Total size of structures directory
du -sh structures/

# Database file size only
du -sh structures/database.db

# Per-structure directory sizes
du -sh structures/*/
```

### Backup Database

#### Quick Backup

```bash
# Backup with date stamp
cp structures/database.db structures/database_backup_$(date +%Y%m%d).db

# Example: creates database_backup_20250115.db
```

#### Scheduled Backups

```bash
# Add to crontab for daily backups at 2 AM
crontab -e

# Add line (replace /path/to/OptiMat-Chat with the actual repo path):
0 2 * * * cd /path/to/OptiMat-Chat && cp structures/database.db structures/database_backup_$(date +\%Y\%m\%d).db
```

#### Backup to External Location

```bash
# Backup to external drive
cp structures/database.db /mnt/backup/optimat-alloys/database_$(date +%Y%m%d).db

# Backup with compression
tar -czf structures_backup_$(date +%Y%m%d).tar.gz structures/
```

### Query Database Stats

The structure DB is an ASE-format SQLite file. Custom fields like `calculator_name` are stored in ASE's `key_value_pairs` (not as plain SQL columns), and properties like the elastic tensor live inside the binary `data` blob. Use the `ase` CLI or a short Python snippet rather than raw SQL.

**ASE CLI** (recommended for quick counts):

```bash
# Total number of structures
ase db structures/database.db --count

# Filter by calculator (key=value syntax)
ase db structures/database.db calculator_name=orb-v3-conservative-inf-omat --count

# List a few rows with selected columns
ase db structures/database.db --columns=id,formula,calculator_name -L 10
```

**Python** (for fields inside the `data` blob, e.g. elastic properties):

```python
from ase.db import connect
db = connect("structures/database.db")

# Count per calculator
from collections import Counter
print(Counter(row.key_value_pairs.get("calculator_name", "Unknown") for row in db.select()))

# Count structures that have an elastic stiffness tensor
n = sum(1 for row in db.select() if "elastic_stiffness_tensor_voigt_GPa" in row.data)
print(f"Structures with elastic tensor: {n}")
```

**Schema introspection** (only if you really want raw SQL):

```bash
sqlite3 structures/database.db ".schema"
sqlite3 structures/database.db ".tables"
```

### Database Integrity Check

```bash
# Check for corruption
sqlite3 structures/database.db "PRAGMA integrity_check;"

# Expected output: "ok"

# If corrupted, restore from backup
cp structures/database_backup_20250115.db structures/database.db
```

## Disk Space Monitoring

### Check Project Size

Run from the repo root:

```bash
# Total project size
du -sh .

# Breakdown by directory
du -h --max-depth=1 . | sort -h
```

### Check Structure Storage

```bash
# Total size of all structures
du -sh structures/

# Size per structure (sorted by size)
du -sh structures/*/ | sort -h

# Find largest structures
du -sh structures/*/ | sort -h | tail -10
```

### Find Large Files

```bash
# Find large trajectory files (>10 MB)
find structures/ -name "*.traj" -size +10M

# Find large image files (>5 MB)
find structures/ -name "*.png" -size +5M

# Find all files larger than 10 MB
find structures/ -type f -size +10M -exec ls -lh {} \;
```

### Disk Space Alerts

Run from the repo root:

```bash
# Check available disk space
df -h .

# Alert if less than 10 GB free
AVAILABLE=$(df . | tail -1 | awk '{print $4}')
if [ $AVAILABLE -lt 10485760 ]; then
  echo "WARNING: Less than 10 GB free!"
fi
```

## Cleaning Up

### Remove Old Visualizations

Structure visualizations can be regenerated on demand using the `generate_report` tool.

```bash
# Remove all PNG images (will be regenerated when needed)
find structures/*/  -name "*.png" -delete

# Count before deletion
find structures/ -name "*.png" | wc -l

# Remove with confirmation
find structures/ -name "*.png" -exec rm -i {} \;
```

**Safe**: Images are regenerated automatically when using `generate_report`.

### Remove Trajectory Files

Trajectory files contain relaxation history but are not needed after structure is relaxed.

```bash
# Remove all trajectory files (keeps only final structures)
find structures/ -name "*.traj" -delete

# Count before deletion
find structures/ -name "*.traj" | wc -l

# Check size that would be freed
du -sh $(find structures/ -name "*.traj")
```

**Warning**: Trajectory files cannot be regenerated. Only delete if you don't need relaxation history.

### Remove Temporary Files

```bash
# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# Remove Jupyter checkpoint files
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null

# Remove MacOS files (if applicable)
find . -name ".DS_Store" -delete
```

### Archive Old Structures

```bash
# Create archive of structures older than 90 days
find structures/ -type d -mtime +90 -exec tar -czf old_structures_$(date +%Y%m%d).tar.gz {} +

# Verify archive
tar -tzf old_structures_20250115.tar.gz

# Remove archived structures (DANGEROUS - verify archive first!)
# find structures/ -type d -mtime +90 -exec rm -rf {} +
```

## Performance Monitoring

### Monitor Memory Usage

```bash
# During Chainlit session
ps aux | grep "chainlit run" | awk '{print $2, $4, $6}'
# Columns: PID, %MEM, RSS (memory in KB)

# Track over time
watch -n 5 'ps aux | grep "chainlit run" | grep -v grep'
```

### Monitor GPU Usage

```bash
# NVIDIA GPU monitoring
nvidia-smi

# Continuous monitoring
watch -n 1 nvidia-smi

# Memory usage only
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Monitor Disk I/O

```bash
# Install iotop (if needed)
sudo apt-get install iotop

# Monitor I/O by process
sudo iotop -o

# Check write operations
iostat -x 5
```

## Log Management

### Chainlit Logs

Chainlit logs are typically written to stdout/stderr. Capture them for debugging:

```bash
# Run with log capture
chainlit run run_chat.py 2>&1 | tee chainlit.log

# View last 50 lines
tail -50 chainlit.log

# Search for errors
grep -i error chainlit.log

# Search for specific structure ID
grep "structure_id=123" chainlit.log
```

### Python Logs

```bash
# Enable Python logging (add to run_chat.py)
import logging
logging.basicConfig(level=logging.DEBUG, filename='debug.log')

# View logs
tail -f debug.log

# Filter by module
grep "src.core.calculators" debug.log
```

## Reference Data Management

Reference energies and lattice constants live in `data/reference/`, one JSON file per calculator (see `src/storage/cache.py`).

### Check Reference Data Status

```bash
# List reference data files
ls -lh data/reference/energies_per_atom*.json data/reference/lattice_constants*.json

# Which calculators have reference data?
ls data/reference/energies_per_atom*.json | sed 's|.*/energies_per_atom_||;s|\.json$||'
```

### Regenerate Reference Data

```bash
# Delete the reference files for the calculator you want to regenerate
rm data/reference/energies_per_atom_orb_v3_conservative_inf_omat.json
rm data/reference/lattice_constants_orb_v3_conservative_inf_omat.json

# Start Chainlit — missing reference data is regenerated on demand
chainlit run run_chat.py
```

**Warning**: Regeneration is expensive — order of magnitude 117 elements × 5 structures, ~8–16 hours per calculator on a single GPU. If you have a backup, restore it instead.

### Backup Reference Data

```bash
# Backup all reference data
tar -czf reference_data_backup_$(date +%Y%m%d).tar.gz data/reference/

# Restore
tar -xzf reference_data_backup_20250115.tar.gz
```

## Common Issues

### Issue: Port Already in Use

**Symptom**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Kill process using the port
kill -9 $(lsof -ti:8000)

# Try different port
chainlit run run_chat.py --port 8001
```

### Issue: CUDA Out of Memory

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Restart the Chainlit session — `torch.cuda.empty_cache()` doesn't release memory held by stale references; a fresh process is the simplest fix.
2. Reduce **Default Supercell Size** in the Chainlit settings panel (Small/48 → Medium/512 → Large/2048). Large supercells with backprop calculators (`orb-v3-conservative-inf-omat`, NequIP) are the most memory-hungry.
3. Switch to a smaller calculator: e.g. `mace-omat-0-small` instead of `-medium`, or `orb-v3-direct-20-omat` (direct forces) instead of the conservative variant.
4. Set `TORCH_COMPILE_DISABLE=1` in `.env` if you're hitting OOM during the first compile pass on CUDA 12.4.

### Issue: Database Locked

**Symptom**: `sqlite3.OperationalError: database is locked`

**Solution**:
```bash
# Close all connections to database
# Check for processes accessing database
lsof structures/database.db

# Kill processes if necessary
kill -9 <PID>
```

### Issue: Stale Calculator Cache

**Symptom**: Changes to calculator not taking effect

**Solution**: `clear_cache()` is an instance method — instantiate `CalculatorManager` first, or simply restart the Chainlit session (the cache lives in process memory, not on disk).

```python
# In a Python session
from src.core.calculators import CalculatorManager
mgr = CalculatorManager()
mgr.clear_cache()
```

## Preventive Maintenance

### Weekly Tasks

- [ ] Check disk space: `df -h`
- [ ] Backup database: `cp structures/database.db structures/database_backup_$(date +%Y%m%d).db`
- [ ] Check for stale processes: `ps aux | grep chainlit`

### Monthly Tasks

- [ ] Clean up old visualizations: `find structures/ -name "*.png" -mtime +30 -delete`
- [ ] Review and archive old structures
- [ ] Update dependencies: `conda update --all` or `pip list --outdated`
- [ ] Check for OptiMat Alloys updates: `git pull`

### Quarterly Tasks

- [ ] Full backup to external storage
- [ ] Review disk space trends
- [ ] Update documentation
- [ ] Regenerate reference data with updated calculators (if needed)

## See Also

- [Configuration Guide](CONFIGURATION.md) - Configuration troubleshooting
