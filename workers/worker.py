import structlog
from redis import Redis
from rq import Worker, Queue

from api.config import get_settings
from api.services.alerts import raise_alert_sync

log = structlog.get_logger()


def alert_exception_handler(job, exc_type, exc_value, tb) -> bool:
    """Fire a developer alert for any failed RQ job.

    Returning True lets RQ's default handling (FailedJobRegistry) run as before.
    The alert dispatcher itself is excluded to prevent alert-about-alert recursion.
    """
    if job.func_name == "workers.alert_job.run":
        log.error("alert.dispatcher_crashed", job_id=job.id, error=str(exc_value))
        return True
    raise_alert_sync(
        severity="error",
        source="worker",
        event=f"job.failed:{job.func_name}",
        detail=str(exc_value),
        context={"job_id": job.id, "args": [str(a) for a in job.args]},
    )
    return True


def main():
    settings = get_settings()
    conn = Redis.from_url(settings.REDIS_URL)
    queues = [Queue("invoice-jobs", connection=conn)]
    worker = Worker(queues, connection=conn, exception_handlers=[alert_exception_handler])
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
