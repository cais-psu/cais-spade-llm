#!/bin/sh
set -eu

route_robot() {
  ROBOT_IP="$1"
  SRC_IP="$2"

  IFACE="$(
    /usr/sbin/ip -o -4 addr show \
      | /usr/bin/awk -v ip="$SRC_IP" '$4 ~ ("^" ip "/") {print $2; exit}'
  )"
  [ -n "$IFACE" ] || return 1

  /usr/sbin/ip route replace "$ROBOT_IP/32" dev "$IFACE" src "$SRC_IP"
  /usr/sbin/ip route get "$ROBOT_IP" \
    | /usr/bin/awk -v iface="$IFACE" -v src="$SRC_IP" '
        $0 ~ (" dev " iface " ") && $0 ~ (" src " src "([[:space:]]|$)") {
          found = 1
        }
        END { exit found ? 0 : 1 }
      '
}

i=0
while [ "$i" -lt 20 ]; do
  ur5e_ready=false
  xarm6_ready=false

  if route_robot 192.168.1.172 192.168.1.220; then
    ur5e_ready=true
  fi
  if route_robot 192.168.1.240 192.168.1.230; then
    xarm6_ready=true
  fi
  if [ "$ur5e_ready" = true ] && [ "$xarm6_ready" = true ]; then
    exit 0
  fi

  i=$((i + 1))
  /bin/sleep 0.5
done

echo "Could not install the xArm6 and UR5e host routes within 10 seconds." >&2
exit 1
