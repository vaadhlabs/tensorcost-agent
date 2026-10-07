"""
AWS monitoring module for GPU instances and cost tracking
"""

import boto3
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json

logger = logging.getLogger(__name__)

class AWSMonitor:
    def __init__(self, config_or_regions: Union[Dict[str, Any], List[str], None] = None):
        # Use IAM roles instead of access keys for better security
        # This will automatically use the EC2 instance profile or IAM role
        session = boto3.Session()

        # Verify we have credentials (either from IAM role or environment)
        try:
            credentials = session.get_credentials()
            if not credentials:
                raise Exception("No AWS credentials found. Please use IAM role or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY")
            logger.info("✅ AWS credentials found - using IAM role or environment variables")
        except Exception as e:
            logger.error(f"❌ AWS credentials error: {e}")
            raise

        regions: Optional[List[str]] = None
        if isinstance(config_or_regions, dict):
            cfg_region = config_or_regions.get('region')
            if cfg_region:
                regions = [cfg_region] if isinstance(cfg_region, str) else list(cfg_region)
        elif isinstance(config_or_regions, list):
            regions = config_or_regions

        self.session = session
        self._regions = regions or self._get_regions()
        self._regional_ec2_clients = {}

        # Use the first region for default clients
        primary_region = self._regions[0] if self._regions else 'us-east-1'
        self.ec2_client = session.client('ec2', region_name=primary_region)
        self.cloudwatch = session.client('cloudwatch', region_name=primary_region)
        self.ce_client = session.client('ce', region_name='us-east-1')  # Cost Explorer is global
        self.sagemaker_client = session.client('sagemaker', region_name=primary_region)

    def _get_regions(self) -> List[str]:
        """Get list of regions to monitor."""
        env_regions = os.getenv('AWS_REGIONS', '').strip()
        if env_regions:
            return [r.strip() for r in env_regions.split(',') if r.strip()]
        return [os.getenv('AWS_REGION', 'us-east-1')]

    def _ec2_client_for_region(self, region: str):
        """Get or create an EC2 client for the given region."""
        if region not in self._regional_ec2_clients:
            self._regional_ec2_clients[region] = self.session.client('ec2', region_name=region)
        return self._regional_ec2_clients[region]
        
    def get_gpu_instances(self) -> List[Dict[str, Any]]:
        """Get all EC2 GPU instances in any state (running, stopped, terminated, etc.) across all regions"""
        try:
            gpu_instance_types = [
                'p2.xlarge', 'p2.8xlarge', 'p2.16xlarge',
                'p3.2xlarge', 'p3.8xlarge', 'p3.16xlarge', 'p3dn.24xlarge',
                'p4d.xlarge', 'p4d.24xlarge',
                # P5 (H100) / P5e/P5en (H200)
                'p5.48xlarge',
                'p5e.48xlarge',
                'p5en.48xlarge',
                # P6 Blackwell (B200 / GB200 UltraServers)
                'p6-b200.48xlarge',
                'u-p6e-gb200x36',
                'u-p6e-gb200x72',
                # P4de
                'p4de.24xlarge',
                'g4dn.xlarge', 'g4dn.2xlarge', 'g4dn.4xlarge', 'g4dn.8xlarge', 'g4dn.12xlarge', 'g4dn.16xlarge',
                'g5.xlarge', 'g5.2xlarge', 'g5.4xlarge', 'g5.8xlarge', 'g5.12xlarge', 'g5.16xlarge', 'g5.24xlarge', 'g5.48xlarge',
                # G6 (L4)
                'g6.xlarge', 'g6.2xlarge', 'g6.4xlarge', 'g6.8xlarge', 'g6.12xlarge', 'g6.16xlarge', 'g6.24xlarge', 'g6.48xlarge',
                # G6e (L40S)
                'g6e.xlarge', 'g6e.2xlarge', 'g6e.4xlarge', 'g6e.8xlarge', 'g6e.12xlarge', 'g6e.16xlarge', 'g6e.24xlarge', 'g6e.48xlarge',
                # Trn1 (Trainium)
                'trn1.2xlarge', 'trn1.32xlarge', 'trn1n.32xlarge',
                # Inf2 (Inferentia2)
                'inf2.xlarge', 'inf2.8xlarge', 'inf2.24xlarge', 'inf2.48xlarge',
            ]

            instances = []

            # Query each region
            for region in self._regions:
                try:
                    ec2_client = self._ec2_client_for_region(region)
                    response = ec2_client.describe_instances(
                        Filters=[
                            {'Name': 'instance-state-name', 'Values': [
                                'running', 'stopped', 'stopping', 'shutting-down', 'terminated'
                            ]},
                            {'Name': 'instance-type', 'Values': gpu_instance_types}
                        ]
                    )

                    for reservation in response['Reservations']:
                        for instance in reservation['Instances']:
                            az = instance['Placement']['AvailabilityZone']
                            instance_data = {
                                'instance_id': instance['InstanceId'],
                                'instance_type': instance['InstanceType'],
                                'state': instance['State']['Name'],
                                'launch_time': instance['LaunchTime'].isoformat(),
                                'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])},
                                'availability_zone': az,
                                'region': region,
                                'cloud_provider': 'aws',
                                'gpu_count': self._get_gpu_count(instance['InstanceType']),
                                'gpu_type': self._get_gpu_type(instance['InstanceType'])
                            }
                            instances.append(instance_data)
                except Exception as e:
                    logger.error(f"Error getting GPU instances from region {region}: {e}")
                    continue

            running = sum(1 for i in instances if i['state'] == 'running')
            non_running = len(instances) - running
            logger.info(f"Found {len(instances)} GPU instances ({running} running, {non_running} other states) across {len(self._regions)} region(s)")
            return instances

        except Exception as e:
            logger.error(f"Error getting GPU instances: {e}")
            return []
    
    def get_gpu_utilization(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return no utilization metrics — AWS/EC2 CloudWatch GPUUtilization is not used.

        Remote EC2 GPU util comes from on-host NVML or DCGM via main.py's
        nvml_sampler / dcgm path after instance inventory is sent. CloudWatch
        GPUUtilization is absent on most instance types and duplicates host telemetry.
        """
        if instances:
            logger.debug(
                "AWS monitor skipping CloudWatch GPUUtilization for %d instance(s); "
                "utilization is sourced from on-host NVML/DCGM",
                len(instances),
            )
        return []
    
    def get_cost_data(self) -> List[Dict[str, Any]]:
        """Get cost data from Cost Explorer"""
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            response = self.ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date,
                    'End': end_date
                },
                Granularity='DAILY',
                Metrics=['BlendedCost'],
                GroupBy=[
                    {
                        'Type': 'DIMENSION',
                        'Key': 'SERVICE'
                    },
                    {
                        'Type': 'DIMENSION',
                        'Key': 'INSTANCE_TYPE'
                    }
                ]
            )
            
            cost_data = []
            for result in response['ResultsByTime']:
                for group in result['Groups']:
                    if 'EC2' in group['Keys'][0]:  # Only EC2 costs
                        cost_data.append({
                            'date': result['TimePeriod']['Start'],
                            'service': group['Keys'][0],
                            'instance_type': group['Keys'][1],
                            'cost': float(group['Metrics']['BlendedCost']['Amount']),
                            'currency': group['Metrics']['BlendedCost']['Unit']
                        })
            
            return cost_data
            
        except Exception as e:
            logger.error(f"Error getting cost data: {e}")
            return []
    
    def _get_gpu_count(self, instance_type: str) -> int:
        """Get GPU count for instance type"""
        gpu_counts = {
            'p2.xlarge': 1, 'p2.8xlarge': 8, 'p2.16xlarge': 16,
            'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p3dn.24xlarge': 8,
            'p4d.xlarge': 1, 'p4d.24xlarge': 8,
            'p5.48xlarge': 8,
            'p5e.48xlarge': 8,
            'p5en.48xlarge': 8,
            'p6-b200.48xlarge': 8,
            'u-p6e-gb200x36': 36,
            'u-p6e-gb200x72': 72,
            'p4de.24xlarge': 8,
            'g4dn.xlarge': 1, 'g4dn.2xlarge': 1, 'g4dn.4xlarge': 1,
            'g4dn.8xlarge': 1, 'g4dn.12xlarge': 4, 'g4dn.16xlarge': 1,
            'g5.xlarge': 1, 'g5.2xlarge': 1, 'g5.4xlarge': 1, 'g5.8xlarge': 1,
            'g5.12xlarge': 4, 'g5.16xlarge': 1, 'g5.24xlarge': 4, 'g5.48xlarge': 8,
            'g6.xlarge': 1, 'g6.2xlarge': 1, 'g6.4xlarge': 1, 'g6.8xlarge': 1, 'g6.12xlarge': 4, 'g6.16xlarge': 1, 'g6.24xlarge': 4, 'g6.48xlarge': 8,
            'g6e.xlarge': 1, 'g6e.2xlarge': 1, 'g6e.4xlarge': 1, 'g6e.8xlarge': 1, 'g6e.12xlarge': 4, 'g6e.16xlarge': 1, 'g6e.24xlarge': 4, 'g6e.48xlarge': 8,
            'trn1.2xlarge': 1, 'trn1.32xlarge': 16, 'trn1n.32xlarge': 16,
            'inf2.xlarge': 1, 'inf2.8xlarge': 1, 'inf2.24xlarge': 6, 'inf2.48xlarge': 12,
        }
        return gpu_counts.get(instance_type, 0)
    
    def _get_gpu_type(self, instance_type: str) -> str:
        """Get GPU type for instance type"""
        itype = instance_type.lower()
        if itype.startswith('p2'):
            return 'K80'
        elif itype.startswith('p3'):
            return 'V100'
        elif 'gb200' in itype or itype.startswith('u-p6e'):
            return 'GB200'
        elif itype.startswith('p6'):
            return 'B200'
        elif itype.startswith('p5en') or itype.startswith('p5e'):
            return 'H200'
        elif itype.startswith('p5'):
            return 'H100'
        elif itype.startswith('p4de'):
            return 'A100-80GB'
        elif instance_type.startswith('p4'):
            return 'A100'
        elif instance_type.startswith('g4dn'):
            return 'T4'
        elif instance_type.startswith('g5'):
            return 'A10G'
        elif instance_type.startswith('g6e'):
            return 'L40S'
        elif instance_type.startswith('g6'):
            return 'L4'
        elif instance_type.startswith('trn1'):
            return 'Trainium'
        elif instance_type.startswith('inf2'):
            return 'Inferentia2'
        else:
            return 'Unknown'

    # ── Recommendation Methods ────────────────────────────────────────

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get RI and Savings Plan purchase recommendations from Cost Explorer.
        Runs on a slow schedule (every 6h), NOT in the monitoring hot path.
        Adds delays between API calls to respect AWS CE rate limits (5 req/s).
        """
        recommendations = []

        try:
            ri_recs = self._get_ri_recommendations()
            recommendations.extend(ri_recs)
        except Exception as e:
            logger.warning(f"Could not fetch RI recommendations: {e}")

        time.sleep(2)  # Rate limit buffer between CE API calls

        try:
            sp_recs = self._get_sp_recommendations()
            recommendations.extend(sp_recs)
        except Exception as e:
            logger.warning(f"Could not fetch Savings Plan recommendations: {e}")

        logger.info(f"Found {len(recommendations)} cost recommendations")
        return recommendations

    def _get_ri_recommendations(self) -> List[Dict[str, Any]]:
        """Get Reserved Instance purchase recommendations"""
        recs = []
        for term in ['ONE_YEAR', 'THREE_YEARS']:
            try:
                response = self.ce_client.get_reservation_purchase_recommendation(
                    Service='Amazon Elastic Compute Cloud - Compute',
                    LookbackPeriodInDays='THIRTY_DAYS',
                    TermInYears=term,
                    PaymentOption='NO_UPFRONT'
                )
                for detail in response.get('Recommendations', []):
                    for item in detail.get('RecommendationDetails', []):
                        instance_type = item.get('InstanceDetails', {}).get('EC2InstanceDetails', {}).get('InstanceType', 'unknown')
                        current_cost = float(item.get('AverageNormalizedUnitsUsedPerHour', 0)) * 730
                        ri_cost = float(item.get('EstimatedMonthlyOnDemandCost', 0)) - float(item.get('EstimatedMonthlySavingsAmount', 0))
                        savings = float(item.get('EstimatedMonthlySavingsAmount', 0))
                        savings_pct = float(item.get('EstimatedMonthlySavingsPercentage', 0))

                        recs.append({
                            'instance_type': instance_type,
                            'recommendation_type': 'ri',
                            'term': '1_year' if term == 'ONE_YEAR' else '3_year',
                            'payment_option': 'no_upfront',
                            'current_monthly_cost': float(item.get('EstimatedMonthlyOnDemandCost', 0)),
                            'recommended_monthly_cost': ri_cost,
                            'potential_monthly_savings': savings,
                            'potential_savings_pct': savings_pct,
                            'usage_hours_per_month': float(item.get('AverageUtilization', 0)) * 730 / 100,
                            'recommended_count': int(item.get('RecommendedNumberOfInstancesToPurchase', 0)),
                            'cloud_provider': 'aws'
                        })
            except Exception as e:
                logger.debug(f"RI recommendation error for {term}: {e}")
            time.sleep(1)  # Rate limit between CE API calls

        return recs

    def _get_sp_recommendations(self) -> List[Dict[str, Any]]:
        """Get Savings Plan purchase recommendations"""
        recs = []
        try:
            response = self.ce_client.get_savings_plans_purchase_recommendation(
                SavingsPlansType='COMPUTE_SP',
                LookbackPeriodInDays='THIRTY_DAYS',
                TermInYears='ONE_YEAR',
                PaymentOption='NO_UPFRONT'
            )

            for detail in response.get('SavingsPlansPurchaseRecommendation', {}).get('SavingsPlansPurchaseRecommendationDetails', []):
                current_cost = float(detail.get('CurrentAverageHourlyOnDemandSpend', 0)) * 730
                sp_cost = float(detail.get('EstimatedAverageUtilization', 0)) / 100 * float(detail.get('HourlyCommitmentToPurchase', 0)) * 730
                savings = float(detail.get('EstimatedMonthlySavingsAmount', 0))

                recs.append({
                    'instance_type': 'compute_sp',
                    'recommendation_type': 'savings_plan',
                    'term': '1_year',
                    'payment_option': 'no_upfront',
                    'current_monthly_cost': current_cost,
                    'recommended_monthly_cost': current_cost - savings,
                    'potential_monthly_savings': savings,
                    'potential_savings_pct': float(detail.get('EstimatedSavingsPercentage', 0)),
                    'usage_hours_per_month': 730,
                    'hourly_commitment': float(detail.get('HourlyCommitmentToPurchase', 0)),
                    'cloud_provider': 'aws'
                })
        except Exception as e:
            logger.debug(f"SP recommendation error: {e}")

        return recs

    # ── Spot Instance Intelligence ─────────────────────────────────────

    # On-demand hourly pricing (USD) for GPU instance types
    _ON_DEMAND_PRICES = {
        'p2.xlarge': 0.900, 'p2.8xlarge': 7.200, 'p2.16xlarge': 14.400,
        'p3.2xlarge': 3.060, 'p3.8xlarge': 12.240, 'p3.16xlarge': 24.480, 'p3dn.24xlarge': 31.212,
        'p4d.24xlarge': 32.773,
        'p5.48xlarge': 98.32,
        'p5e.48xlarge': 113.88,
        'p5en.48xlarge': 125.44,
        'p6-b200.48xlarge': 98.84,
        'u-p6e-gb200x36': 380.95,
        'u-p6e-gb200x72': 761.90,
        'p4de.24xlarge': 40.97,
        'g4dn.xlarge': 0.526, 'g4dn.2xlarge': 0.752, 'g4dn.4xlarge': 1.204,
        'g4dn.8xlarge': 2.176, 'g4dn.12xlarge': 3.912, 'g4dn.16xlarge': 4.352,
        'g5.xlarge': 1.006, 'g5.2xlarge': 1.212, 'g5.4xlarge': 1.624,
        'g5.8xlarge': 2.448, 'g5.12xlarge': 5.672, 'g5.16xlarge': 4.096,
        'g5.24xlarge': 8.144, 'g5.48xlarge': 16.288,
        'g6.xlarge': 0.805, 'g6.2xlarge': 0.978, 'g6.4xlarge': 1.323, 'g6.8xlarge': 2.013, 'g6.12xlarge': 4.602, 'g6.16xlarge': 3.397, 'g6.24xlarge': 6.675, 'g6.48xlarge': 13.35,
        'g6e.xlarge': 1.171, 'g6e.2xlarge': 1.517, 'g6e.4xlarge': 2.209, 'g6e.8xlarge': 3.593, 'g6e.12xlarge': 8.358, 'g6e.16xlarge': 6.362, 'g6e.24xlarge': 12.089, 'g6e.48xlarge': 24.178,
    }

    def get_spot_pricing(self) -> List[Dict[str, Any]]:
        """Get current spot prices for GPU instance types and compare with on-demand."""
        spot_data = []
        try:
            gpu_types = list(self._ON_DEMAND_PRICES.keys())
            since = datetime.now() - timedelta(hours=6)

            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=gpu_types,
                ProductDescriptions=['Linux/UNIX'],
                StartTime=since,
                MaxResults=500
            )

            # Get latest price per (instance_type, AZ)
            latest = {}
            for item in response.get('SpotPriceHistory', []):
                key = (item['InstanceType'], item['AvailabilityZone'])
                if key not in latest or item['Timestamp'] > latest[key]['Timestamp']:
                    latest[key] = item

            for (itype, az), item in latest.items():
                spot_price = float(item['SpotPrice'])
                on_demand = self._ON_DEMAND_PRICES.get(itype, 0)
                savings_pct = ((on_demand - spot_price) / on_demand * 100) if on_demand > 0 else 0

                spot_data.append({
                    'instance_type': itype,
                    'availability_zone': az,
                    'spot_price_hourly': spot_price,
                    'on_demand_price_hourly': on_demand,
                    'spot_monthly': round(spot_price * 730, 2),
                    'on_demand_monthly': round(on_demand * 730, 2),
                    'savings_pct': round(savings_pct, 1),
                    'cloud_provider': 'aws'
                })

            logger.info(f"Collected spot pricing for {len(spot_data)} instance type/AZ combos")
        except Exception as e:
            logger.warning(f"Error fetching spot pricing: {e}")

        return spot_data

    def get_spot_candidates(self, instances: List[Dict[str, Any]], spot_pricing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify running instances that are good spot candidates based on tags."""
        candidates = []
        # Build spot price lookup: instance_type -> best spot price
        best_spot = {}
        for sp in spot_pricing:
            itype = sp['instance_type']
            if itype not in best_spot or sp['spot_price_hourly'] < best_spot[itype]['spot_price_hourly']:
                best_spot[itype] = sp

        spot_tags = {'batch', 'training', 'dev', 'test', 'experiment', 'ci', 'staging'}

        for inst in instances:
            if inst.get('state') != 'running':
                continue

            itype = inst.get('instance_type', '')
            if itype not in best_spot:
                continue

            tags = inst.get('tags', {})
            tag_values = {v.lower() for v in tags.values()}
            tag_keys = {k.lower() for k in tags.keys()}

            # Check if instance is a spot candidate based on tags
            is_candidate = False
            reason = ''

            if tags.get('fault_tolerant', '').lower() in ('true', 'yes', '1'):
                is_candidate = True
                reason = 'Tagged as fault-tolerant'
            elif tags.get('workload_type', '').lower() in spot_tags:
                is_candidate = True
                reason = f"Workload type: {tags.get('workload_type')}"
            elif tag_values & spot_tags:
                is_candidate = True
                reason = 'Tags suggest non-production workload'
            elif 'environment' in tag_keys and tags.get('environment', '').lower() in ('dev', 'staging', 'test'):
                is_candidate = True
                reason = f"Environment: {tags.get('environment')}"

            if is_candidate:
                sp = best_spot[itype]
                on_demand_monthly = sp['on_demand_monthly']
                spot_monthly = sp['spot_monthly']

                candidates.append({
                    'instance_id': inst['instance_id'],
                    'instance_type': itype,
                    'recommendation_type': 'spot',
                    'term': 'on_demand',
                    'payment_option': 'spot',
                    'current_monthly_cost': on_demand_monthly,
                    'recommended_monthly_cost': spot_monthly,
                    'potential_monthly_savings': round(on_demand_monthly - spot_monthly, 2),
                    'potential_savings_pct': sp['savings_pct'],
                    'cloud_provider': 'aws',
                    'source_data': {
                        'reason': reason,
                        'instance_id': inst['instance_id'],
                        'spot_price_hourly': sp['spot_price_hourly'],
                        'on_demand_price_hourly': sp['on_demand_price_hourly'],
                        'availability_zone': sp['availability_zone'],
                        'tags': tags
                    }
                })

        logger.info(f"Found {len(candidates)} spot candidates out of {len(instances)} instances")
        return candidates

    # ── Action Methods ──────────────────────────────────────────────────

    def stop_instance(self, instance_id: str, region: str = None) -> Dict[str, Any]:
        """Stop an EC2 instance"""
        ec2_client = self._ec2_client_for_region(region) if region else self.ec2_client
        result = ec2_client.stop_instances(InstanceIds=[instance_id])
        state = result['StoppingInstances'][0]['CurrentState']['Name']
        logger.info(f"Stopped instance {instance_id} -> {state}")
        return {'instance_id': instance_id, 'state': state}

    def start_instance(self, instance_id: str, region: str = None) -> Dict[str, Any]:
        """Start an EC2 instance"""
        ec2_client = self._ec2_client_for_region(region) if region else self.ec2_client
        result = ec2_client.start_instances(InstanceIds=[instance_id])
        state = result['StartingInstances'][0]['CurrentState']['Name']
        logger.info(f"Started instance {instance_id} -> {state}")
        return {'instance_id': instance_id, 'state': state}

    def resize_instance(self, instance_id: str, target_type: str, region: str = None) -> Dict[str, Any]:
        """Resize an EC2 instance (stop → modify → start)"""
        ec2_client = self._ec2_client_for_region(region) if region else self.ec2_client
        # Stop
        ec2_client.stop_instances(InstanceIds=[instance_id])
        waiter = ec2_client.get_waiter('instance_stopped')
        waiter.wait(InstanceIds=[instance_id])

        # Modify
        ec2_client.modify_instance_attribute(
            InstanceId=instance_id,
            InstanceType={'Value': target_type}
        )

        # Start
        ec2_client.start_instances(InstanceIds=[instance_id])
        logger.info(f"Resized instance {instance_id} to {target_type}")
        return {'instance_id': instance_id, 'new_type': target_type}

    def restart_instance(self, instance_id: str, region: str = None) -> Dict[str, Any]:
        """Reboot an EC2 instance"""
        ec2_client = self._ec2_client_for_region(region) if region else self.ec2_client
        ec2_client.reboot_instances(InstanceIds=[instance_id])
        logger.info(f"Rebooted instance {instance_id}")
        return {'instance_id': instance_id, 'action': 'reboot'}

    def terminate_instance(self, instance_id: str, region: str = None) -> Dict[str, Any]:
        """Terminate an EC2 instance"""
        ec2_client = self._ec2_client_for_region(region) if region else self.ec2_client
        ec2_client.terminate_instances(InstanceIds=[instance_id])
        logger.info(f"Terminated instance {instance_id}")
        return {'instance_id': instance_id, 'action': 'terminate'}
