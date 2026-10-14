"""Linux Landlock + seccomp. There is deliberately no unsandboxed execution mode."""
import ctypes as c
import ctypes.util
import errno
import os
from pathlib import Path
import platform
import sys


def restrict(read_paths, write_paths):
    if sys.platform != 'linux' or platform.machine() not in ('x86_64', 'aarch64'):
        raise RuntimeError('Round22 generated code requires Linux Landlock ABI >=3 and libseccomp')
    libc=c.CDLL(None,use_errno=True)
    sec_library=ctypes.util.find_library('seccomp') or 'libseccomp.so.2'
    libc.syscall.restype=c.c_long
    abi=libc.syscall(444,0,0,1)
    if abi<3:raise RuntimeError('Landlock ABI >=3 required; no candidate executed')
    # EXECUTE, WRITE_FILE, READ_FILE, READ_DIR, REMOVE, MAKE, REFER, TRUNCATE.
    handled=(1<<15)-1
    class Ruleset(c.Structure):_fields_=[('handled_access_fs',c.c_uint64)]
    class Rule(c.Structure):
        _pack_=1
        _fields_=[('allowed_access',c.c_uint64),('parent_fd',c.c_int32)]
    attrs=Ruleset(handled);fd=libc.syscall(444,c.byref(attrs),c.sizeof(attrs),0)
    if fd<0:raise OSError(c.get_errno(),'landlock_create_ruleset')
    try:
        for path,write in [(p,False) for p in read_paths]+[(p,True) for p in write_paths]:
            path=Path(path).resolve()
            if not path.exists():continue
            access=sum(1<<i for i in (1,2,3,4,5,7,8,13,14)) if write else ((1<<2)|(1<<3) if path.is_dir() else 1<<2)
            if write and not path.is_dir():access=(1<<1)|(1<<2)|(1<<14)
            parent=os.open(path,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=Rule(access,parent)
                if libc.syscall(445,fd,1,c.byref(rule),0)<0:raise OSError(c.get_errno(),'landlock_add_rule')
            finally:os.close(parent)
        if libc.prctl(38,1,0,0,0)!=0:raise OSError(c.get_errno(),'no_new_privs')
        if libc.syscall(446,fd,0)<0:raise OSError(c.get_errno(),'landlock_restrict_self')
    finally:os.close(fd)
    # Load before confinement to support environments where libseccomp is outside /usr/lib.
    sec=c.CDLL(sec_library,use_errno=True)
    sec.seccomp_init.argtypes=[c.c_uint32];sec.seccomp_init.restype=c.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes=[c.c_char_p];sec.seccomp_syscall_resolve_name.restype=c.c_int
    sec.seccomp_rule_add.argtypes=[c.c_void_p,c.c_uint32,c.c_int,c.c_uint]
    sec.seccomp_load.argtypes=[c.c_void_p];sec.seccomp_release.argtypes=[c.c_void_p]
    ctx=sec.seccomp_init(0x7fff0000)
    if not ctx:raise RuntimeError('seccomp_init failed')
    denied=0x00050000|errno.EPERM
    try:
        for name in ('socket','socketpair','connect','bind','listen','accept','accept4','execve','execveat','fork','vfork',
                     'ptrace','process_vm_readv','process_vm_writev','mount','umount2','pivot_root','unshare','setns','bpf',
                     'open_by_handle_at','name_to_handle_at','io_uring_setup','io_uring_enter','io_uring_register',
                     'kill','tkill','pidfd_open','pidfd_getfd','pidfd_send_signal','setuid','setgid','setresuid','setresgid','capset',
                     'mknod','mknodat','setrlimit','prlimit64'):
            nr=sec.seccomp_syscall_resolve_name(name.encode())
            if nr>=0 and sec.seccomp_rule_add(ctx,denied,nr,0)!=0:raise RuntimeError('seccomp deny failed: '+name)
        class Arg(c.Structure):_fields_=[('arg',c.c_uint),('op',c.c_int),('a',c.c_uint64),('b',c.c_uint64)]
        sec.seccomp_rule_add_array.argtypes=[c.c_void_p,c.c_uint32,c.c_int,c.c_uint,c.POINTER(Arg)]
        clone3=sec.seccomp_syscall_resolve_name(b'clone3')
        if clone3>=0 and sec.seccomp_rule_add(ctx,0x00050000|errno.ENOSYS,clone3,0)!=0:raise RuntimeError('seccomp clone3 fallback failed')
        # Only CLONE_THREAD clones may be created; no child process or namespaces.
        clone=sec.seccomp_syscall_resolve_name(b'clone')
        cmp=Arg(0,7,0x10000,0)
        if sec.seccomp_rule_add_array(ctx,denied,clone,1,c.byref(cmp))!=0:raise RuntimeError('seccomp clone filter failed')
        prctl=sec.seccomp_syscall_resolve_name(b'prctl')
        cmp=Arg(0,4,1,0)
        if sec.seccomp_rule_add_array(ctx,denied,prctl,1,c.byref(cmp))!=0:raise RuntimeError('seccomp parent-death filter failed')
        if sec.seccomp_load(ctx)!=0:raise RuntimeError('seccomp_load failed')
    finally:sec.seccomp_release(ctx)


def runtime_paths():
    """Libraries are readable, main repository, home, other processes and labels are not."""
    paths=[sys.base_prefix+'/lib',sys.prefix+'/lib','/usr/lib','/usr/lib64','/lib','/lib64','/usr/local/cuda','/usr/local/nvidia','/usr/local/lib',
           '/etc/ld.so.cache','/etc/localtime','/etc/passwd','/proc/self','/proc/thread-self','/proc/meminfo','/proc/cpuinfo',
           '/proc/stat','/proc/sys/kernel','/sys/devices/system/cpu','/sys/bus/pci','/proc/driver/nvidia']
    paths += [str(p) for p in Path('/dev').glob('nvidia*')]
    return paths
