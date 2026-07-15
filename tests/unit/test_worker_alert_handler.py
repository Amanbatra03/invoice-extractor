from unittest.mock import MagicMock, patch


def _job(func_name="workers.extract_job.run"):
    job = MagicMock()
    job.func_name = func_name
    job.id = "rq-job-1"
    job.args = ("inv-1", "job-1")
    return job


def test_handler_alerts_on_job_failure():
    from workers import worker
    job = _job()
    with patch.object(worker, "raise_alert_sync") as mock_alert:
        result = worker.alert_exception_handler(job, RuntimeError, RuntimeError("boom"), None)
    assert result is True  # RQ default handling (FailedJobRegistry) still runs
    mock_alert.assert_called_once_with(
        severity="error",
        source="worker",
        event="job.failed:workers.extract_job.run",
        detail="boom",
        context={"job_id": "rq-job-1", "args": ["inv-1", "job-1"]},
    )


def test_handler_skips_alert_job_to_avoid_recursion():
    from workers import worker
    job = _job(func_name="workers.alert_job.run")
    with patch.object(worker, "raise_alert_sync") as mock_alert:
        result = worker.alert_exception_handler(job, RuntimeError, RuntimeError("boom"), None)
    assert result is True
    mock_alert.assert_not_called()
