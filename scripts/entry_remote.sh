#!/usr/bin/expect
set user root
set ipaddress dgx
set passwd afish123
set timeout 30
set port 2232

spawn ssh $user@$ipaddress -p $port
expect {
    "*password:" { send "$passwd\r" }
    "yes/no" { send "yes\r";exp_continue }
}
expect "root@*"
send "cd /home/yuanzhaolin/VAE-AKF\r"
interact
