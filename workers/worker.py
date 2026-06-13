from redis import Redis
from rq import Worker, Queue

from api.config import get_settings


def main():
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("invoice-jobs", connection=conn)]
    worker = Worker(queues, connection=conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
