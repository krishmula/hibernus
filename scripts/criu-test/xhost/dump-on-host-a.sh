set -u
D=/var/tmp/xhost; sudo rm -rf $D; sudo mkdir -p $D/img; sudo chown ubuntu $D
cp ~/counter.py $D/
sudo unshare --pid --fork setsid python3 $D/counter.py 30 1 > $D/out.log 2>&1 < /dev/null &
sleep 2; PID=$(pgrep -x python3 | head -1); echo "host A pid $PID: $(grep NSpid /proc/$PID/status)"
until grep -q "^step 7 " $D/out.log; do sleep 0.5; done
sudo criu dump -t $PID -D $D/img --shell-job -v4 -o $D/img/dump.log; echo "criu dump rc=$?"
sleep 1; kill -0 $PID 2>/dev/null && echo STILL-ALIVE || echo "original gone on host A"
echo "images: $(du -sh $D/img | cut -f1)"; tail -2 $D/out.log
sudo tar -C $D -czf ~/xhost.tgz img out.log; sudo chown ubuntu ~/xhost.tgz; ls -l ~/xhost.tgz
