# This is what we want
from mlrun.artifacts.model import update_model


def promote_champion(
    project,
    model,
):
    models = project.list_models(labels={"status": "champion"})
    if not models:
        update_model(
            model_artifact=model, labels={"status": "champion"}, write_spec_copy=False
        )
    else:
        cha_tag = models[-1].to_dict()["metadata"]["tag"]
        raise KeyError(f"Champion model already exists: {cha_tag}")


def promote_challenger(
    project,
    model,
):
    """
    This is especially useful for canary deployment
    """
    chal_models = project.list_models(labels={"status": "challenger"})
    champ_models = project.list_models(labels={"status": "champion"})
    if champ_models:
        champion_tags = [x.to_dict()["metadata"]["tag"] for x in champ_models]
        challenger_model_tag = model.to_dict()["metadata"]["tag"]
        if challenger_model_tag in champion_tags:
            raise KeyError(
                f"Challenger model is a currently champion: {challenger_model_tag}"
            )
    if not chal_models:
        update_model(
            model_artifact=model, labels={"status": "challenger"}, write_spec_copy=False
        )
    else:
        chall_tag = chal_models[-1].to_dict()["metadata"]["tag"]
        raise KeyError(f"Challenger model already exists: {chall_tag}")


def promote_challenger_demote_champion(
    project,
):
    """
    This is used in canary deployment. Have to create challenger model first, for integrity check
    """
    # Check champion exists
    champ_models = project.list_models(labels={"status": "champion"})
    if not champ_models:
        raise KeyError(f"Champion model does not exists")

    # Check challenger exists
    chall_models = project.list_models(labels={"status": "challenger"})
    if not chall_models:
        raise KeyError(f"Challenger model does not exist")

    # Promote, demote
    champion_tag = champ_models[-1].to_dict()["metadata"]["tag"]
    update_model(
        model_artifact=chall_models[-1],
        labels={"status": "champion", "replaced": champion_tag},
        write_spec_copy=False,
    )

    update_model(
        model_artifact=champ_models[-1],
        labels={"status": "standby"},
        write_spec_copy=False,
    )


def demote_challenger(
    project,
):
    """
    This is especially useful for canary deployment
    """
    chal_models = project.list_models(labels={"status": "challenger"})

    if chal_models:
        chal_model = chal_model[-1]
        update_model(
            model_artifact=chal_model,
            labels={"status": "standby"},
            write_spec_copy=False,
        )
    else:
        raise KeyError(f"Challenger model does not exist")
