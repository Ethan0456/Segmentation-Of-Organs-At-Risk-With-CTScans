import subprocess

class Download():
    def __init__(self, download_dir='asperaDownloads'):
        self.download_dir = download_dir
    
    def download(self):
        # List of commands to run to install dependencies
        commands = [
            "sudo apt-get install ruby-dev",
            "sudo apt install libtool libffi-dev ruby ruby-dev make",
            "sudo apt install libzmq3-dev libczmq-dev",
            "gem install ffi-rzmq rake",
            "gem install specific_install",
            "gem specific_install https://github.com/SciRuby/iruby",
            "iruby register",
            "gem install net-protocol -v 0.1.2",
            "gem install net-smtp -v 0.3.0",
            "gem install aspera-cli",
            "ascli conf ascp install"
            f"!ascli faspex package recv --url=https://faspex.cancerimagingarchive.net/aspera/faspex --username=none --password=none --to-folder={self.download_dir} https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/259?passcode=caf2906b2acd7345a7f275ba96475214dc0c1bf9#"
        ]

        # Run each command
        for command in commands:
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {command}")
                print(e)
