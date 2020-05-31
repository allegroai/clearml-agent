from __future__ import print_function

from six.moves import input
from pyhocon import ConfigFactory, ConfigMissingException
from pathlib2 import Path
from six.moves.urllib.parse import urlparse

from trains_agent.backend_api.session import Session
from trains_agent.backend_api.session.defs import ENV_HOST
from trains_agent.backend_config.defs import LOCAL_CONFIG_FILES


description = """
Please create new trains credentials through the profile page in your trains web app (e.g. https://demoapp.trains.allegro.ai/profile)
In the profile page, press "Create new credentials", then press "Copy to clipboard".

Paste copied configuration here: 
"""

def_host = 'http://localhost:8080'
try:
    def_host = ENV_HOST.get(default=def_host) or def_host
except Exception:
    pass

host_description = """
Editing configuration file: {CONFIG_FILE}
Enter the url of the trains-server's Web service, for example: {HOST}
""".format(
    CONFIG_FILE=LOCAL_CONFIG_FILES[0],
    HOST=def_host,
)


def main():
    print('TRAINS-AGENT setup process')
    conf_file = Path(LOCAL_CONFIG_FILES[0]).absolute()
    if conf_file.exists() and conf_file.is_file() and conf_file.stat().st_size > 0:
        print('Configuration file already exists: {}'.format(str(conf_file)))
        print('Leaving setup, feel free to edit the configuration file.')
        return

    print(description, end='')
    sentinel = ''
    parse_input = '\n'.join(iter(input, sentinel))
    credentials = None
    api_server = None
    web_server = None
    # noinspection PyBroadException
    try:
        parsed = ConfigFactory.parse_string(parse_input)
        if parsed:
            # Take the credentials in raw form or from api section
            credentials = get_parsed_field(parsed, ["credentials"])
            api_server = get_parsed_field(parsed, ["api_server", "host"])
            web_server = get_parsed_field(parsed, ["web_server"])
    except Exception:
        credentials = credentials or None
        api_server = api_server or None
        web_server = web_server or None

    while not credentials or set(credentials) != {"access_key", "secret_key"}:
        print('Could not parse credentials, please try entering them manually.')
        credentials = read_manual_credentials()

    print('Detected credentials key=\"{}\" secret=\"{}\"'.format(credentials['access_key'],
                                                                 credentials['secret_key'][0:4] + "***"))
    web_input = True
    if web_server:
        host = input_url('WEB Host', web_server)
    elif api_server:
        web_input = False
        host = input_url('API Host', api_server)
    else:
        print(host_description)
        host = input_url('WEB Host', '')

    parsed_host = verify_url(host)
    api_host, files_host, web_host = parse_host(parsed_host, allow_input=True)

    # on of these two we configured
    if not web_input:
        web_host = input_url('Web Application Host', web_host)
    else:
        api_host = input_url('API Host', api_host)

    files_host = input_url('File Store Host', files_host)

    print('\nTRAINS Hosts configuration:\nWeb App: {}\nAPI: {}\nFile Store: {}\n'.format(
        web_host, api_host, files_host))

    retry = 1
    max_retries = 2
    while retry <= max_retries:  # Up to 2 tries by the user
        if verify_credentials(api_host, credentials):
            break
        retry += 1
        if retry < max_retries + 1:
            credentials = read_manual_credentials()
    else:
        print('Exiting setup without creating configuration file')
        return

    # get GIT User/Pass for cloning
    print('Enter git username for repository cloning (leave blank for SSH key authentication): [] ', end='')
    git_user = input()
    if git_user.strip():
        print('Enter password for user \'{}\': '.format(git_user), end='')
        git_pass = input()
        print('Git repository cloning will be using user={} password={}'.format(git_user, git_pass))
    else:
        git_user = None
        git_pass = None

    # get extra-index-url for pip installations
    extra_index_urls = []
    print('\nEnter additional artifact repository (extra-index-url) to use when installing python packages '
          '(leave blank if not required):', end='')
    index_url = input().strip()
    while index_url:
        extra_index_urls.append(index_url)
        print('Another artifact repository? (enter another url or leave blank if done):', end='')
        index_url = input().strip()
    if len(extra_index_urls):
        print("The following artifact repositories will be added:\n\t- {}".format("\n\t- ".join(extra_index_urls)))

    # noinspection PyBroadException
    try:
        conf_folder = Path(__file__).parent.absolute() / '..' / 'backend_api' / 'config' / 'default'
        default_conf = ''
        for conf in ('agent.conf', 'sdk.conf', ):
            conf_file_section = conf_folder / conf
            with open(str(conf_file_section), 'rt') as f:
                default_conf += conf.split('.')[0] + ' '
                default_conf += f.read()
            default_conf += '\n'
    except Exception:
        print('Error! Could not read default configuration file')
        return
    # noinspection PyBroadException
    try:
        with open(str(conf_file), 'wt') as f:
            header = '# TRAINS-AGENT configuration file\n' \
                     'api {\n' \
                     '    api_server: %s\n' \
                     '    web_server: %s\n' \
                     '    files_server: %s\n' \
                     '    # Credentials are generated using the webapp, %s/profile\n' \
                     '    # Override with os environment: TRAINS_API_ACCESS_KEY / TRAINS_API_SECRET_KEY\n' \
                     '    credentials {"access_key": "%s", "secret_key": "%s"}\n' \
                     '}\n\n' % (api_host, web_host, files_host,
                                web_host, credentials['access_key'], credentials['secret_key'])
            f.write(header)
            git_credentials = '# Set GIT user/pass credentials\n' \
                              '# leave blank for GIT SSH credentials\n' \
                              'agent.git_user=\"{}\"\n' \
                              'agent.git_pass=\"{}\"\n' \
                              '\n'.format(git_user or '', git_pass or '')
            f.write(git_credentials)
            extra_index_str = '# extra_index_url: ["https://allegroai.jfrog.io/trainsai/api/pypi/public/simple"]\n' \
                              'agent.package_manager.extra_index_url= ' \
                              '[\n{}\n]\n\n'.format("\n".join(map("\"{}\"".format, extra_index_urls)))
            f.write(extra_index_str)
            f.write(default_conf)
    except Exception:
        print('Error! Could not write configuration file at: {}'.format(str(conf_file)))
        return

    print('\nNew configuration stored in {}'.format(str(conf_file)))
    print('TRAINS-AGENT setup completed successfully.')


def parse_host(parsed_host, allow_input=True):
    if parsed_host.netloc.startswith('demoapp.'):
        # this is our demo server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demoapi.', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapp.', 'demofiles.',
                                                                             1) + parsed_host.path
    elif parsed_host.netloc.startswith('app.'):
        # this is our application server
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('app.', 'api.', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('app.', 'files.', 1) + parsed_host.path
    elif parsed_host.netloc.startswith('demoapi.'):
        print('{} is the api server, we need the web server. Replacing \'demoapi.\' with \'demoapp.\''.format(
            parsed_host.netloc))
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demoapp.', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('demoapi.', 'demofiles.',
                                                                             1) + parsed_host.path
    elif parsed_host.netloc.startswith('api.'):
        print('{} is the api server, we need the web server. Replacing \'api.\' with \'app.\''.format(
            parsed_host.netloc))
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'app.', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace('api.', 'files.', 1) + parsed_host.path
    elif parsed_host.port == 8008:
        print('Port 8008 is the api port. Replacing 8080 with 8008 for Web application')
        api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8080', 1) + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8008', ':8081', 1) + parsed_host.path
    elif parsed_host.port == 8080:
        api_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8008', 1) + parsed_host.path
        web_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
        files_host = parsed_host.scheme + "://" + parsed_host.netloc.replace(':8080', ':8081', 1) + parsed_host.path
    elif allow_input:
        api_host = ''
        web_host = ''
        files_host = ''
        if not parsed_host.port:
            print('Host port not detected, do you wish to use the default 8080 port n/[y]? ', end='')
            replace_port = input().lower()
            if not replace_port or replace_port == 'y' or replace_port == 'yes':
                api_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8008' + parsed_host.path
                web_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8080' + parsed_host.path
                files_host = parsed_host.scheme + "://" + parsed_host.netloc + ':8081' + parsed_host.path
            elif not replace_port or replace_port.lower() == 'n' or replace_port.lower() == 'no':
                web_host = input_host_port("Web", parsed_host)
                api_host = input_host_port("API", parsed_host)
                files_host = input_host_port("Files", parsed_host)
        if not api_host:
            api_host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
    else:
        raise ValueError("Could not parse host name")

    return api_host, files_host, web_host


def verify_credentials(api_host, credentials):
    """check if the credentials are valid"""
    # noinspection PyBroadException
    try:
        print('Verifying credentials ...')
        if api_host:
            Session(api_key=credentials['access_key'], secret_key=credentials['secret_key'], host=api_host)
            print('Credentials verified!')
            return True
        else:
            print("Can't verify credentials")
            return False
    except Exception:
        print('Error: could not verify credentials: key={} secret={}'.format(
            credentials.get('access_key'), credentials.get('secret_key')))
        return False


def get_parsed_field(parsed_config, fields):
    """
    Parsed the value from web profile page, 'copy to clipboard' option
    :param parsed_config: The parsed value from the web ui
    :type parsed_config: Config object
    :param fields: list of values to parse, will parse by the list order
    :type fields: List[str]
    :return: parsed value if found, None else
    """
    try:
        return parsed_config.get("api").get(fields[0])
    except ConfigMissingException:  # fallback - try to parse the field like it was in web older version
        if len(fields) == 1:
            return parsed_config.get(fields[0])
        elif len(fields) == 2:
            return parsed_config.get(fields[1])
        else:
            return None


def read_manual_credentials():
    print('Enter user access key: ', end='')
    access_key = input()
    print('Enter user secret: ', end='')
    secret_key = input()
    return {"access_key": access_key, "secret_key": secret_key}


def input_url(host_type, host=None):
    while True:
        print('{} configured to: [{}] '.format(host_type, host), end='')
        parse_input = input()
        if host and (not parse_input or parse_input.lower() == 'yes' or parse_input.lower() == 'y'):
            break
        parsed_host = verify_url(parse_input) if parse_input else None
        if parse_input and parsed_host:
            host = parsed_host.scheme + "://" + parsed_host.netloc + parsed_host.path
            break
    return host


def input_host_port(host_type, parsed_host):
    print('Enter port for {} host '.format(host_type), end='')
    replace_port = input().lower()
    return parsed_host.scheme + "://" + parsed_host.netloc + (':{}'.format(replace_port) if replace_port else '') + \
           parsed_host.path


def verify_url(parse_input):
    try:
        if not parse_input.startswith('http://') and not parse_input.startswith('https://'):
            # if we have a specific port, use http prefix, otherwise assume https
            if ':' in parse_input:
                parse_input = 'http://' + parse_input
            else:
                parse_input = 'https://' + parse_input
        parsed_host = urlparse(parse_input)
        if parsed_host.scheme not in ('http', 'https'):
            parsed_host = None
    except Exception:
        parsed_host = None
        print('Could not parse url {}\nEnter your trains-server host: '.format(parse_input), end='')
    return parsed_host


if __name__ == '__main__':
    main()
