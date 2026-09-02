# MetaSTC-J Remote Server Instructions

## Server connection

- SSH host: `4090`
- Host name: `219.216.64.250`
- User: `root`
- Port: `32835`
- Server password: `1234` (sensitive; do not expose in logs or commit publicly)
- Keepalive: `ServerAliveInterval 60`, `ServerAliveCountMax 3`

Example SSH configuration:

```sshconfig
Host 4090
    HostName 219.216.64.250
    User root
    Port 32835
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Remote workspace

- Project directory: `/workspace/MetaSTC-J`
- Run project commands from `/workspace/MetaSTC-J`.
- One-shot command example: `ssh 4090 'cd /workspace/MetaSTC-J && <command>'`

## Runtime and experiments

- Check the available Python installation on the server before running experiments; do not assume the local macOS Python path exists remotely.
- Verify `sys.executable` and the PyTorch installation before experiments.
- Resolve data, model, and log paths relative to `/workspace/MetaSTC-J`.
- Use `tmux` for long-running experiments so jobs survive SSH disconnects.
- Confirm the current directory with `pwd` before editing or launching jobs.
