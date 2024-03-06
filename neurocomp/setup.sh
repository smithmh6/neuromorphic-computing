INSTALL_DIR=/opt/intel/openvino
INSTALL_URL=https://storage.openvinotoolkit.org/repositories/openvino/packages/2022.2/linux/l_openvino_toolkit_debian9_arm_2022.2.0.7713.af16ea1d79a_armhf.tgz
INSTALL_FILE=l_openvino_toolkit_debian9_arm_2022.2.0.7713.af16ea1d79a_armhf.tgz
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR
wget -c $INSTALL_URL
tar xf $INSTALL_FILE --strip 1 -C $INSTALL_DIR
sudo usermod -aG users "$(whoami)"
source $INSTALL_DIR/setupvars.sh
echo "source $INSTALL_DIR/setupvars.sh" >> ~/.bashrc
cd $INSTALL_DIR/install_dependencies/
sh install_NCS_udev_rules.sh
