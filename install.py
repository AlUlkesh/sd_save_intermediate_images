import launch

if not launch.is_installed("picologging"):
    launch.run_pip("install picologging", "picologging requirement for Save intermediate images")
