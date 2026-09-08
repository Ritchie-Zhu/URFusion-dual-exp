#!/usr/bin/env bash
setsid bash /root/autodl-tmp/URFusion-main/metrics_save/m3fd_ten_methods/wait_then_m3fd_pipeline.sh \
	>> /root/autodl-tmp/URFusion-main/metrics_save/m3fd_ten_methods/wait_pipeline.nohup 2>&1 < /dev/null &
echo "M3FD watcher PID=$!"
echo "Log: metrics_save/m3fd_ten_methods/wait_pipeline.log"
