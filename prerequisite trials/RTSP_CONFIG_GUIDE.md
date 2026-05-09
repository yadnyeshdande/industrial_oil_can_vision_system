# CP Plus Camera RTSP Configuration Guide

## Step 1: Access Camera Web Interface

1. Open your browser
2. Go to: **http://192.168.1.249**
3. Login with:
   - Username: `admin`
   - Password: `Pass_123`

## Step 2: Find RTSP Settings

Look for one of these menu paths (varies by camera model):

### Path 1: Via Configuration
- **Configuration → Network → RTSP**
- Check if RTSP is **Enabled**
- Note the RTSP port (usually 554)
- Check if there's a stream path configured

### Path 2: Via Video/Streaming
- **Video → Stream Settings**
- **Streaming → Main Stream / Sub Stream**
- Look for RTSP URL or stream path
- Verify RTSP is enabled

### Path 3: Via Live View
- **Live View → Stream Settings**
- Find RTSP stream configuration
- Test stream availability

## Step 3: Important Settings to Check

### Enable RTSP
- [ ] RTSP must be **Enabled** (not disabled)
- Check the checkbox if it's disabled

### RTSP Port
- Default: **554**
- If different, note it (e.g., 8554, 8000)

### Stream Type
- Main stream (high quality)
- Sub stream (lower quality)

### Authentication
- Check if authentication is required
- Your URL includes: `rtsp://admin:Pass_123@...`

### Stream Path Examples
The path might be shown as:
- `/live`
- `/stream1`
- `/stream`
- `cam/realmonitor?channel=1`
- `/ch0/main/av_stream`
- `/Streaming/Channels/101`

## Step 4: Construct Your RTSP URL

Once you find the settings, your URL will be:
```
rtsp://admin:Pass_123@192.168.1.249:554/STREAM_PATH
```

Replace `STREAM_PATH` with what you found in the settings.

## Step 5: Test the URL

1. Save the URL you created
2. Run:
```bash
python camera_discovery.py "rtsp://admin:Pass_123@192.168.1.249:554/YOUR_PATH"
```

3. If it works, use this URL in your camera process

## Troubleshooting

### RTSP Not Enabled
- Go to Network settings
- Find RTSP option
- Click **Enable**
- Apply and restart camera

### Multiple Stream Paths
Some cameras have:
- **Main Stream**: `channel=1&subtype=0`
- **Sub Stream**: `channel=1&subtype=1`

Try both to see which one works.

### Factory Reset (Last Resort)
If settings are corrupted:
1. Find the **Reset** button in Network → Advanced
2. Select **Reset to Factory Defaults**
3. Restart camera
4. Login with default credentials (usually admin/admin or admin/123456)

## Common CP Plus Default Credentials
- Username: `admin`
- Password: `admin`
- Or: `admin` / `123456`
- Or: `admin` / `Pass_123`

If your credentials don't work after reset, try alternatives above.
