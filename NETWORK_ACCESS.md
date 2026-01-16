# Network Access Guide

Access the EchoSee Voice UI from any device on your local network.

## Your Network Information

**Host Machine IP**: `192.168.100.203`

## Access URLs

### From Your Local Machine:
- Web UI: http://localhost:9003/voice
- Voice API: http://localhost:9004

### From Other Devices on Network:
- Web UI: http://192.168.100.203:9003/voice
- Voice API: http://192.168.100.203:9004

## Setup Instructions

### 1. Start the Servers

The servers are already configured to accept network connections:

```bash
./scripts/launch_voice_ui.sh
```

You'll see:
```
Voice API: http://localhost:9004
Web UI: http://localhost:9003
  - Network:  http://192.168.100.203:9003
```

### 2. Access from Another Device

**On your phone/tablet/another computer:**

1. Make sure the device is on the same WiFi network
2. Open a browser and go to: `http://192.168.100.203:9003/voice`
3. Grant microphone permissions when prompted
4. Start talking!

## How It Works

The frontend automatically detects if you're accessing from the network and adjusts the API URLs:

- **localhost access** → Uses `http://localhost:9004` for API
- **network access** → Uses `http://192.168.100.203:9004` for API

This happens automatically - no configuration needed!

## Firewall Configuration

If you can't connect from other devices, you may need to allow the ports through your firewall:

### Linux (UFW):
```bash
sudo ufw allow 9003/tcp
sudo ufw allow 9004/tcp
```

### Linux (firewalld):
```bash
sudo firewall-cmd --add-port=9003/tcp --permanent
sudo firewall-cmd --add-port=9004/tcp --permanent
sudo firewall-cmd --reload
```

### macOS:
Firewall usually allows local network connections by default.
If blocked, go to System Preferences → Security & Privacy → Firewall → Firewall Options

### Windows:
Windows Firewall usually prompts you automatically.
If needed: Control Panel → Windows Defender Firewall → Allow an app

## Testing Connectivity

### From another device, test the API:
```bash
curl http://192.168.100.203:9004/health
```

Should return:
```json
{"status":"healthy"}
```

### Test from the host machine:
```bash
# Test localhost
curl http://localhost:9004/health

# Test network interface
curl http://192.168.100.203:9004/health
```

## Common Issues

### Can't access from phone/tablet:

**1. Check both devices are on same network:**
```bash
# On host machine
ip addr show | grep inet
```

**2. Ping from other device:**
```bash
ping 192.168.100.203
```

**3. Check servers are running:**
```bash
# Should show both ports
netstat -tuln | grep -E "9003|9004"
```

**4. Temporarily disable firewall to test:**
```bash
# Linux
sudo ufw disable
# Remember to re-enable: sudo ufw enable
```

### Network IP Changed:

If your machine's IP changes (DHCP), update the URL:

```bash
# Find new IP
hostname -I | awk '{print $1}'

# Access with new IP
http://<NEW_IP>:9003/voice
```

### Safari Microphone Blocking:

**⚠️ Safari on iOS/macOS blocks microphone access over HTTP network connections for security.**

When accessing from Safari at `http://192.168.100.203:9003/voice`, you'll see:
> "Safari requires HTTPS for microphone access over network. Please access from the host machine at http://localhost:9003/voice instead."

**Solutions:**

1. **Access from host machine** at http://localhost:9003/voice (recommended)
2. **Use Chrome or Firefox** on the mobile device (works over HTTP)
3. **Set up HTTPS** (advanced - requires SSL certificate):
   ```bash
   # Generate self-signed cert
   openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
   # Then configure Next.js to use HTTPS
   ```

### Other Browser Issues:

The app uses HTTP (not HTTPS). Most browsers allow microphone access over HTTP only for:
- `localhost` and `127.0.0.1`
- Chrome/Firefox may allow local network (192.168.x.x)
- Safari blocks all non-localhost HTTP

## Mobile Browser Tips

### iOS (Safari):
- Grant microphone permission when prompted
- If blocked, go to Settings → Safari → Microphone → Allow for the site

### Android (Chrome):
- Grant microphone permission when prompted
- If blocked, tap the lock icon in address bar → Permissions

## Performance Notes

- **Latency**: Expect slightly higher latency over WiFi vs localhost
- **Bandwidth**: Audio upload ~100KB per recording
- **Concurrent Users**: The backend can handle multiple simultaneous requests

## Security Considerations

**Important**: This setup allows anyone on your local network to:
- Access the voice assistant
- See conversation history
- Use your OpenAI API quota

**For production use:**
1. Add authentication (login system)
2. Use HTTPS with valid certificate
3. Implement rate limiting
4. Consider using a reverse proxy (nginx)
5. Set up proper CORS policies

**For local network use:** This setup is fine!

## Advanced: Static IP

To avoid IP changes, configure a static IP:

### Linux (Ubuntu):
Edit `/etc/netplan/*.yaml`:
```yaml
network:
  ethernets:
    eth0:
      addresses: [192.168.100.203/24]
      gateway4: 192.168.100.1
      nameservers:
        addresses: [8.8.8.8, 8.8.4.4]
```

Apply:
```bash
sudo netplan apply
```

### Router DHCP Reservation:
Better option - configure your router to always assign the same IP to your machine's MAC address.

## Troubleshooting Commands

```bash
# Check what's listening on ports
sudo netstat -tlnp | grep -E "9003|9004"

# Check if ports are accessible
nc -zv 192.168.100.203 9003
nc -zv 192.168.100.203 9004

# View server logs
# Voice API logs will show in the terminal where you ran the script

# Check firewall status
sudo ufw status  # Linux (UFW)
sudo firewall-cmd --list-all  # Linux (firewalld)
```

## Quick Test Script

Save as `test_network_access.sh`:
```bash
#!/bin/bash
IP="192.168.100.203"

echo "Testing network access..."
echo ""
echo "1. Testing Voice API:"
curl -s http://$IP:9004/health && echo " ✓ Voice API OK" || echo " ✗ Voice API FAILED"

echo ""
echo "2. Testing Next.js:"
curl -s http://$IP:9003/ > /dev/null && echo " ✓ Next.js OK" || echo " ✗ Next.js FAILED"

echo ""
echo "3. Your access URL:"
echo "   http://$IP:9003/voice"
```

Run it:
```bash
chmod +x test_network_access.sh
./test_network_access.sh
```

Now you can access EchoSee from anywhere on your local network! 🎤
