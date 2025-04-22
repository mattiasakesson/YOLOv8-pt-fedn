import argparse
import os

import yaml
from fedn.network.clients.fedn_client import ConnectToApiResult, FednClient

from trainer import Trainer




with open(os.path.join("utils", "args.yaml"), errors="ignore") as f:
    params = yaml.safe_load(f)


def main():
    project_url = os.getenv("PROJECT_URL")
    print("project_url: ", project_url)
    client_token = os.getenv("FEDN_AUTH_TOKEN")
    print("client_token: ", client_token)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", default=640, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--local_updates", default=25, type=int)
    args = parser.parse_args()

    trainer = Trainer(args, params)

    fedn_client = FednClient(
        train_callback=trainer.train, validate_callback=trainer.validate
    )

    name = "Tora"

    fedn_client.set_name(name)

    # client_id = str(uuid.uuid4())
    client_id = "214"
    fedn_client.set_client_id(client_id)
    print(client_id)
    controller_config = {
        "name": name,
        "client_id": client_id,
        "package": "local",
        "preferred_combiner": "",
    }

    result, combiner_config = fedn_client.connect_to_api(
        "https://" + project_url + "/", client_token, controller_config
    )

    print("result: ", result)
    print(combiner_config)
    if result != ConnectToApiResult.Assigned:
        print("Failed to connect to API, exiting.")
        exit(1)

    result: bool = fedn_client.init_grpchandler(
        config=combiner_config, client_name=name, token=client_token
    )
    print("result: ", result)
    if not result:
        exit(1)

    fedn_client.run()


if __name__ == "__main__":
    main()
    # training_local()
