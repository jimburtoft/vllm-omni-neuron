#!/bin/bash
# try p5.4xlarge spot across regions/AZs; prints instance id + region on success
for R in us-west-2 us-east-2 ap-northeast-1 eu-west-2 us-east-1; do
  AMI=$(aws ssm get-parameter --region $R --profile permanent --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id --query Parameter.Value --output text 2>/dev/null) || continue
  KEY=OC-moshi-h100-$R
  aws ec2 describe-key-pairs --region $R --profile permanent --key-names $KEY >/dev/null 2>&1 || {
    aws ec2 create-key-pair --region $R --profile permanent --key-name $KEY --key-type ed25519 --query KeyMaterial --output text > ~/.ssh/$KEY.pem && chmod 600 ~/.ssh/$KEY.pem; }
  SG=$(aws ec2 describe-security-groups --region $R --profile permanent --filters Name=group-name,Values=OC-ssh --query 'SecurityGroups[0].GroupId' --output text)
  if [ "$SG" = "None" ]; then
    SG=$(aws ec2 create-security-group --region $R --profile permanent --group-name OC-ssh --description "OC ssh" --query GroupId --output text)
    aws ec2 authorize-security-group-ingress --region $R --profile permanent --group-id $SG --protocol tcp --port 22 --cidr 0.0.0.0/0 >/dev/null
  fi
  for SUB in $(aws ec2 describe-subnets --region $R --profile permanent --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text); do
    OUT=$(aws ec2 run-instances --region $R --profile permanent --image-id $AMI --instance-type p5.4xlarge --key-name $KEY \
      --security-group-ids $SG --subnet-id $SUB \
      --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
      --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=OC-moshi-native-h100}]' \
      --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":300,"VolumeType":"gp3"}}]' \
      --query 'Instances[0].InstanceId' --output text 2>&1)
    if echo "$OUT" | grep -q "^i-"; then echo "OK $R $OUT $KEY"; exit 0; fi
    echo "$R $SUB: $(echo "$OUT" | grep -o 'InsufficientInstanceCapacity\|MaxSpotInstanceCountExceeded\|Unsupported\|InvalidParameter[A-Za-z]*\|SpotMaxPriceTooLow' | head -1)"
  done
done
echo NONE
