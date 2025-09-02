{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3f27c867",
   "metadata": {},
   "outputs": [],
   "source": [
    "import robomaster\n",
    "from robomaster import robot,chassis\n",
    "import math \n",
    "import time"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0391d542",
   "metadata": {},
   "outputs": [],
   "source": [
    "import robomaster\n",
    "from robomaster import robot, chassis\n",
    "import math\n",
    "import time\n",
    "\n",
    "acc_x = 0\n",
    "acc_y = 0\n",
    "\n",
    "def sub_imu_info_handler(imu_info):\n",
    "    global acc_x, acc_y\n",
    "    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_info\n",
    "\n",
    "    #print(\"chassis imu: acc_x:{0}, acc_y:{1}, acc_z:{2}, gyro_x:{3}, gyro_y:{4}, gyro_z:{5}\".format(\n",
    "    #    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z))\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    ep_robot = robot.Robot()\n",
    "    ep_robot.initialize(conn_type=\"ap\")\n",
    "\n",
    "    ep_chassis = ep_robot.chassis\n",
    "    # 订阅底盘IMU信息\n",
    "    ep_chassis.sub_imu(freq=20, callback=sub_imu_info_handler)\n",
    "    a=0\n",
    "    try:\n",
    "        while a<0.6:\n",
    "            # คำนวณมุม\n",
    "            angle = math.atan2(acc_y, acc_x)\n",
    "            angle_deg = math.degrees(angle)\n",
    "            ep_chassis.move(x=0.1, y=0, z=angle_deg, xy_speed=0.5).wait_for_completed()\n",
    "\n",
    "            time.sleep(0.05) # หน่วงเวลาเล็กน้อย\n",
    "            a+=0.1\n",
    "    except KeyboardInterrupt:\n",
    "        ep_chassis.unsub_imu()\n",
    "        ep_robot.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "4b9d9ad6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import robomaster\n",
    "from robomaster import robot, chassis\n",
    "import math\n",
    "import time\n",
    "\n",
    "acc_x = 0\n",
    "acc_y = 0\n",
    "\n",
    "def sub_imu_info_handler(imu_info):\n",
    "    global acc_x,acc_y,angle_deg\n",
    "    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_info\n",
    "    angle = math.atanh(acc_y, acc_x)\n",
    "    angle_deg = math.degrees(angle)\n",
    "    print(angle,angle_deg)\n",
    "    print(acc_x,acc_y,\"angle:\",angle_deg)\n",
    "if __name__ == '__main__':\n",
    "    ep_robot = robot.Robot()\n",
    "    ep_robot.initialize(conn_type=\"ap\")\n",
    "\n",
    "    ep_chassis = ep_robot.chassis\n",
    "    # 订阅底盘IMU信息\n",
    "    \n",
    "    # ep_chassis.move(x=0.1, y=0, z=0, xy_speed=1).wait_for_completed()\n",
    "    # time.sleep(1)\n",
    "    # acc_y_set=acc_y\n",
    "    # acc_x_set=acc_x\n",
    "    \n",
    "    # print(a)\n",
    "    \n",
    "    # if acc_y_set<0:\n",
    "    #     angle_deg=-angle_deg\n",
    "    # print(angle_deg)\n",
    "    ep_chassis.sub_imu(freq=20, callback=sub_imu_info_handler)\n",
    "    ep_chassis.move(x=0.5, y=0, z=0, xy_speed=0.5).wait_for_completed()\n",
    "    \n",
    "    time.sleep(ep_chassis.sub_imu(freq=20, callback=sub_imu_info_handler))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "myrobot",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
