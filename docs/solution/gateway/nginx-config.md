# 配置指令

本节介绍的是 Nginx 配置中的指令说明。

> [!NOTE]
> 本节的配置样例是为了方便您理解 Nginx 配置指令的功能和使用方法，切勿将其直接用于实际生产环境。

## 核心进程指令

```nginx
daemon on;                                              # 以守护进程的方式运行Nginx
pid  logs/nginx.pid;                                    # 主进程ID记录在logs/nginx.pid中
user nobody nobody;                                     # 工作进程运行用户为nobody
load_module "modules/ngx_http_xslt_filter_module.so";   # 加载动态模块ngx_http_xslt_filter_module.so
error_log  logs/error.log debug;                        # 错误日志输出级别为debug
pcre_jit on;                                            # 启用pcre_jit技术
thread_pool default threads=32 max_queue=65536;         # 线程池的线程数为32，等待队列中的最大任务数为65536
timer_resolution 100ms;                                 # 定时器周期为100毫秒
worker_priority -5;                                     # 工作进程系统优先级为-5
worker_processes auto;                                  # 工作进程数由Nginx自动调整
worker_cpu_affinity auto;                               # 工作进程的CPU绑定由Nginx自动调整
worker_rlimit_nofile 65535;                             # 所有工作进程的最大连接数是65535
worker_shutdown_timeout 10s;                            # 工作进程关闭等待时间是10秒
lock_file logs/nginx.lock;                              # 互斥锁文件的位置是logs/nginx.lock

working_directory logs                                  # 工作进程工作目录是logs
debug_points stop;                                      # 调试点模式为stop
worker_rlimit_core 800m;                                # 崩溃文件大小为800MB

events {
    worker_connections 65535;                           # 每个工作进程的最大连接数是65535
    use epoll;                                          # 指定事件模型为epoll
    accept_mutex on;                                    # 启用互斥锁模式的进程调度
    accept_mutex_delay 300ms;                           # 互斥锁模式下进程等待时间为300毫秒
    multi_accept on;                                    # 启用支持多连接
    worker_aio_requests 128;                            # 完成异步操作最大数为128
    debug_connection 192.0.2.0/24;                      # 调试指定连接的IP地址和端口是192.0.2.0/24
}
```

## HTTP 指令

```nginx
http {
    resolver 192.168.2.11 valid=30s;    # 全局域名解析服务器为192.168.2.11，30s更新一次DNS缓存
    resolver_timeout 10s;             # 域名解析超时时间为10s
    variables_hash_max_size 1024;     # Nginx变量的hash表的大小为1024字节
    variables_hash_bucket_size 64;    # Nginx变量的hash表的哈希桶的大小是64字节
    types_hash_max_size 1024;         # MIME类型映射表哈希表的大小为1024字节
    types_hash_bucket_size 64;        # MIME类型映射表哈希桶的大小是64字节

    # 请求解析，HTTP全局有效
    ignore_invalid_headers on;        # 忽略请求头中无效的属性名
    underscores_in_headers on;        # 允许请求头的属性名中有下划线“_”
    client_header_buffer_size 2k;     # 客户请求头缓冲区大小为2KB
    large_client_header_buffers 4 16k;# 超大客户请求头缓冲区大小为64KB
    client_header_timeout  30s;       # 读取客户请求头的超时时间是30s
    request_pool_size 4k;             # 请求池的大小是4K

    merge_slashes on;                 # 当URI中有连续的斜线时做合并处理
    server_tokens off;                # 当返回错误信息时，不显示Nginx服务的版本号信息
    msie_padding on;                  # 当客户端请求出错时，在响应数据中添加注释

    subrequest_output_buffer_size 8k; # 子请求响应报文缓冲区大小为8KB

    lingering_close on;                # Nginx主动关闭连接时启用延迟关闭
    lingering_time 60s;              # 延迟关闭的处理数据的最长时间是60s
    lingering_timeout 5s;              # 延迟关闭的超时时间是5s
    reset_timedout_connection on;       # 当Nginx主动关闭连接而客户端无响应时，在连接超时后进行关闭

    log_not_found on;               　　　# 将未找到文件的错误信息记录到日志中
    log_subrequest on;                # 将子请求的访问日志记录到访问日志中
    error_page 404             /404.html; # 所有请求的404状态码返回404.html文件的数据
    error_page 500 502 503 504 /50x.html; # 所有请求的500、502、503、504状态码返回50×.html文件的数据

    server {
        # 监听本机的 8000 端口，当前服务是 http 指令域的主服务，开启 fastopen 功能并限定最大队列数是 30，拒绝空数据连接，Nginx 工作进程共享 socket 监听端口，当请求阻塞时挂起队列数是 1024 个，当 socket 为保持连接时，开启状态检测功能
        listen *:8000 default_server fastopen=30 deferred reuseport backlog=1024 so_keepalive=on;
        server_name a.nginxbar.com b.nginxtest.net c.nginxbar.com a.nginxbar.com;
        server_names_hash_max_size 1024;  # 服务主机名哈希表大小为1024字节
        server_names_hash_bucket_size 128;# 服务主机名哈希桶大小为128字节


        # 保持链接配置
        keepalive_disable msie6;      # 对MSIE6版本的客户端关闭保持连接机制
        keepalive_requests 1000;      # 保持连接可复用的HTTP连接为1000个
        keepalive_timeout 60s;        # 保持连接空置超时时间为60s
        tcp_nodelay on;               # 当处于保持连接状态时，以最快的方式发送数据


        # 本地文件相关配置
        root /data/website;           # 当前服务对应本地文件访问的根目录是/data/website
        disable_symlinks off;         # 对本地文件路径中的符号链接不做检测

        # 静态文件场景
        location / {
            server_name_in_redirect on; # 在重定向时，拼接服务主机名
            port_in_redirect on;      # 在重定向时，拼接服务主机端口
            if_modified_since exact;  # 当请求头中有if_modified_since属性时，
                                      # 与被请求的本地文件修改时间做精确匹配处理
            etag on;                        # 启用etag功能
            msie_refresh on; # 当客户端是msie时，以添加HTML头信息的方式执行跳转
            open_file_cache max=1000 inactive=20s;# 对被打开文件启用缓存支持，缓存元素数最大为 1000个，不活跃的缓存元素保存20s
            open_file_cache_errors on;       # 对无法找到文件的错误元素也进行缓存
            open_file_cache_min_uses 2;      # 缓存中的元素至少要被访问两次才为活跃
            open_file_cache_valid 60s;       # 每60s对缓存元素与本地文件进行一次检查
        }

        # 上传接口的场景应用
        location /upload {
            alias /data/upload              # 将upload的请求重定位到目录/data/upload
            limit_except GET {              # 对除GET以外的所有方法进行限制
                allow 192.168.100.1;        # 允许192.168.100.1执行所有请求方法
                deny all;                   # 其他IP只允许执行GET方法
            }
            client_max_body_size 200m;         # 允许上传的最大文件大小是200MB
            client_body_buffer_size 16k;       # 上传缓冲区的大小是16KB
            client_body_in_single_buffer on;     # 上传文件完整地保存在临时文件中
            client_body_in_file_only off;        # 不禁用上传缓冲区
            client_body_temp_path /tmp/upload 1 2;# 设置请求体临时文件存储目录
            client_body_timeout 120s;             # 请求体接收超时时间为120s
        }

        # 下载场景应用
        location /download {
            alias /data/upload              # 将download的请求重定位到目录/data/upload
            types { }
            default_type application/octet-stream; # 设置当前目录所有文件默认MIME类型为 application/octet-stream
            try_files $uri @nofile;         # 当文件不存在时，跳转到location @nofile
            sendfile on;                    # 开启零复制文件传输功能
            sendfile_max_chunk 1M;          # 每个sendfile调用的最大传输量为1MB
            tcp_nopush on;                  # 启用最小传输限制功能
            aio on;                         # 启用异步传输
            directio 5M;                    # 当文件大于5MB时以直接读取磁盘方式读取文件
            directio_alignment 4096;        # 与磁盘的文件系统对齐
            output_buffers 4 32k;           # 文件输出的缓冲区为128KB
            limit_rate 1m;                  # 限制下载速度为1MB
            limit_rate_after 2m;            # 当客户端下载速度达到2MB时，进入限速模式
            max_ranges 4096;                # 客户端执行范围读取的最大值是4096B
            send_timeout 20s;               # 客户端引发传输超时时间为20s
            postpone_output 2048;           # 当缓冲区的数据达到2048B时再向客户端发送
            chunked_transfer_encoding on;   # 启用分块传输标识
        }

        location @nofile {
            index nofile.html
        }
        location = /404.html {
            internal;
        }
        location = /50x.html {
            internal;
        }
    }
}
```
