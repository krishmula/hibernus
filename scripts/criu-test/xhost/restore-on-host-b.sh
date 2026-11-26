set -u
D=/var/tmp/xhost; sudo rm -rf $D; sudo mkdir -p $D; sudo tar -C $D -xzf ~/xhost.tgz
cp ~/counter.py $D/ 2>/dev/null
echo "host B before restore: $(hostname), pids in use up to: $(ls /proc | grep -E '^[0-9]+$' | sort -n | tail -1)"
sudo criu restore -D $D/img -d --shell-job -v4 -o $D/img/restore.log; echo "criu restore rc=$?"
sleep 1; PID=$(pgrep -x python3 | head -1); echo "host B pid $PID: $(grep NSpid /proc/$PID/status)"
sleep 25; echo "== out.log tail"; tail -26 $D/out.log
