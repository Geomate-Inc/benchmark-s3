# type: ignore
import boto3
from scenario_a import scenario_A
from scenario_b import scenario_B
from scenario_c import scenario_C


def main():
    scenario_A()
    scenario_B()
    scenario_C()


if __name__ == "__main__":
    main()
