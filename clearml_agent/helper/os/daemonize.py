import os


def daemonize_process(redirect_fd=None):
    """
    Detach a process from the controlling terminal and run it in the background as a daemon.
    """
    assert redirect_fd is None or isinstance(redirect_fd, int)

    # re-spawn in the same directory
    WORKDIR = os.getcwd()

    # The standard I/O file descriptors are redirected to /dev/null by default.
    if hasattr(os, "devnull"):
        devnull = os.devnull
    else:
        devnull = "/dev/null"

    try:
        # Fork a child process so the parent can exit.  This returns control to
        # the command-line or shell.  It also guarantees that the child will not
        # be a process group leader, since the child receives a new process ID
        # and inherits the parent's process group ID.  This step is required
        # to insure that the next call to os.setsid is successful.
        pid = os.fork()
    except OSError as e:
        raise Exception("%s [%d]" % (e.strerror, e.errno))

    if pid == 0:  # The first child.
        # To become the session leader of this new session and the process group
        # leader of the new process group, we call os.setsid().
        # The process is also guaranteed not to have a controlling terminal.
        os.setsid()

        # Is ignoring SIGHUP necessary? (Set handlers for asynchronous events.)
        # import signal
        # signal.signal(signal.SIGHUP, signal.SIG_IGN)

        try:
            # Fork a second child and exit immediately to prevent zombies.  This
            # causes the second child process to be orphaned, making the init
            # process responsible for its cleanup.
            pid = os.fork()  # Fork a second child.
        except OSError as e:
            raise Exception("%s [%d]" % (e.strerror, e.errno))

        if pid == 0:  # The second child.
            # Since the current working directory may be a mounted filesystem, we
            # avoid the issue of not being able to unmount the filesystem at
            # shutdown time by changing it to the root directory.
            os.chdir(WORKDIR)
            # We probably don't want the file mode creation mask inherited from
            # the parent, so we give the child complete control over permissions.
            os.umask(0)
        else:
            # Exit parent (the first child) of the second child.
            os._exit(0)
    else:
        # Exit parent of the first child.
        os._exit(0)

    # notice we count on the fact that we keep all file descriptors open,
    # since we opened then in the parent process, but the daemon process will use them

    # Redirect the standard I/O file descriptors to the specified file /dev/null.
    if redirect_fd is None:
        redirect_fd = os.open(devnull, os.O_RDWR)

    # Duplicate standard input to standard output and standard error.
    # standard output (1), standard error (2)
    os.dup2(redirect_fd, 1)
    os.dup2(redirect_fd, 2)

    return 0
