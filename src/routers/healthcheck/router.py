from fastapi import APIRouter, Request, Response

router = APIRouter()

@router.get("/")
def health_check(request: Request, response: Response):
    return {"status": "OK!"}
