import json, jwt
from flask import Blueprint, request, jsonify, current_app, Response
from flask_restful import Api, Resource # used for REST API building
from datetime import datetime
from auth_middleware import token_required
from model.users import User, Design

user_api = Blueprint('user_api', __name__,
                   url_prefix='/api/users')
titanic_api = Blueprint('titanic_api', __name__,
                    url_prefix='api/titanics')

# API docs https://flask-restful.readthedocs.io/en/latest/api.html
api = Api(user_api)
api2 = Api(titanic_api)

class UserAPI:        
    class _CRUD(Resource):  # User API operation for Create, Read.  THe Update, Delete methods need to be implemeented
        def post(self): # Create method
            ''' Read data for json body '''
            body = request.get_json()
            
            ''' Avoid garbage in, error checking '''
            # validate name
            name = body.get('name')
            if name is None or len(name) < 2:
                return {'message': f'Name is missing, or is less than 2 characters'}, 400
            # validate uid
            uid = body.get('uid')
            if uid is None or len(uid) < 2:
                return {'message': f'User ID is missing, or is less than 2 characters'}, 400
            # look for password and dob
            password = body.get('password')
            dob = body.get('dob')

            ''' #1: Key code block, setup USER OBJECT '''
            uo = User(name=name, 
                      uid=uid,images="Thing")
            
            ''' Additional garbage error checking '''
            # set password if provided
            if password is not None:
                uo.set_password(password)
            # convert to date type
            if dob is not None:
                try:
                    uo.dob = datetime.strptime(dob, '%Y-%m-%d').date()
                except:
                    return {'message': f'Date of birth format error {dob}, must be mm-dd-yyyy'}, 400
            
            ''' #2: Key Code block to add user to database '''
            # create user in database
            user = uo.create()
            print(uo)
            # success returns json of user
            if user:
                return jsonify(user.read())
            # failure returns error
            return {'message': f'Processed {name}, either a format error or User ID {uid} is duplicate'}, 400

        @token_required
        def get(self, current_user): # Read Method
            print("get successful")
            users = User.query.all()    # read/extract all users from database
            json_ready = [user.read() for user in users]  # prepare output in json
            return jsonify(json_ready)  # jsonify creates Flask response object, more specific to APIs than json.dumps
        
        @token_required
        def put(self, current_user):
            body = request.get_json() # get the body of the request
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            uid = body.get('uid')
            name = body.get('name')
            password = body.get('password')
            users = User.query.all()
            for user in users:
                if user.uid == cur_user:
                    if uid == None:
                        uid = user.uid
                    if name == None:
                        name = user.name
                    if password == None:
                        password = user.password
                    user.update(name,uid,password)
            
                
            return f"{user.read()} Updated"
        
        @token_required
        def delete(self, current_user):
        # body = request.get_json()
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: # modified with the and user.id==cur_user so random users can't delete other ppl
                    user.delete()
            return jsonify(user.read())
        
        @token_required
        def patch(self, current_user):
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user:
                    thing = {
                        "id": user.id,
                        "name": user.name,
                        "uid": user.uid,
                        "type": user.type,
                    }
                    return jsonify(thing)
                    
    
    class _DesignCRUD(Resource):  # Design CRUD
        @token_required
        def post(self, current_user): # Create design
            ''' Read data for json body '''
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: # modified with the and user.id==cur_user so random users can't delete other ppl
                    id = user.id
            print("here")
            body = request.get_json()
            name = body.get('name')
            content = body.get('content')
            description = body.get('description')
            type = body.get('type')
            if (type != "public" and type != "private"):
                return {'message': f'Design type must be public or private'}, 400
            do = Design(id=id, type=type, content=content, name=name,description=description)
            design = do.create()
            # success returns json of user
            if design:
                return jsonify(user.read())
        
        @token_required
        def delete(self, current_user):
            body = request.get_json() # get the body of the request
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: # modified with the and user.id==cur_user so random users can't delete other ppl
                    id = user.id
            like = body.get('like')
            dislike = body.get('dislike')
            name = body.get('name')
            if (like != "add") and (dislike != "add") and (like != "remove") and (dislike != "remove"):
                return f"Like/Dislike must be add or remove", 400
            designs = Design.query.all()
            for design in designs:
                if design.userID == id and design.name == name:
                    design.update('','','', (1 if like == "add" else (-1 if like == "remove" else 0)), (1 if dislike == "add" else (-1 if dislike == "remove" else 0)))
                    return f"{design.read()} Updated"
            return f"Cannot locate design", 400
        
        @token_required
        def put(self, current_user):
            body = request.get_json() # get the body of the request
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: # modified with the and user.id==cur_user so random users can't delete other ppl
                    id = user.id
            name = body.get('name')
            content = body.get('content')
            type = body.get('type')
            description = body.get('description')
            designs = Design.query.all()
            for design in designs:
                if design.userID == id and design.name == name:
                    design.update('',content,type,0,0,description)
                    return f"{design.read()} Updated"
            return f"Cannot locate design", 400
        
        @token_required
        def patch(self, current_user):
            body = request.get_json() # get the body of the request
            name = body.get('name')
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid']
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: # modified with the and user.id==cur_user so random users can't delete other ppl
                    id = user.id
            designs = Design.query.all()
            for design in designs:
                if design.userID == id and design.name == name:
                    return jsonify(design.read())
            return f"Cannot locate design", 400

    class _SearchCRUD(Resource):
        # public search of all designs
        def get(self):
            design_return=[]# all designs stored in the database
            designs = Design.query.all()
            for design in designs: # we going through every design
                if(design.read()['Type']=='public'):
                    design_return.append(design.__repr__())
            return jsonify({"Designs":design_return}) # returning designs of all users that are public
        
        # get all private designs
        @token_required
        def put(self, current_user):
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid'] # current user
            users = User.query.all()
            for user in users:
                if user.uid==cur_user: 
                    id = user.id
            designs=Design.query.all() # this is all the designs for the user
            design_return=[]# all designs stored in the database for the user
            for design in designs: # we going through every design
                if design.userID == id:
                    design_return.append(design.__repr__())
            return jsonify({"Designs":design_return}) # returning all the designs of the user        
    class Images(Resource):
        @token_required
        def post(self,current_user):
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid'] # current user
            body=request.get_json()
            base64=body.get("Image")
            users = User.query.all()
            for user in users:
                if user.uid == cur_user:
                    user.updatepfp(base64)
            print("succesful")
        @token_required
        def get(self,current_user):
            print("here")
            token = request.cookies.get("jwt")
            cur_user = jwt.decode(token, current_app.config["SECRET_KEY"], algorithms=["HS256"])['_uid'] # current user
            users = User.query.all()
            for user in users:
                if user.uid == cur_user:
                    # print(type(user))
                    # print(jsonify(user.getProfile()))
                    return jsonify(user.getprofile())
    class _Security(Resource):
        def post(self):
            try:
                body = request.get_json()
                if not body:
                    return {
                        "message": "Please provide user details",
                        "data": None,
                        "error": "Bad request"
                    }, 400
                ''' Get Data '''
                uid = body.get('uid')
                if uid is None:
                    return {'message': f'User ID is missing'}, 400
                password = body.get('password')
                
                ''' Find user '''
                user = User.query.filter_by(_uid=uid).first()
                if user is None or not user.is_password(password):
                    return {'message': f"Invalid user id or password"}, 400
                if user:
                    try:
                        token = jwt.encode(
                            {"_uid": user._uid},
                            current_app.config["SECRET_KEY"],
                            algorithm="HS256"
                        )
                        resp = Response("Authentication for %s successful" % (user._uid))
                        resp.set_cookie("jwt", token,
                                max_age=3600,
                                secure=True,
                                httponly=False,
                                path='/',
                                samesite='None'  # This is the key part for cross-site requests

                                # domain="frontend.com"
                                )
                        return resp
                    except Exception as e:
                        return {
                            "error": "Something went wrong",
                            "message": str(e)
                        }, 500
                return {
                    "message": "Error fetching auth token!",
                    "data": None,
                    "error": "Unauthorized"
                }, 404
            except Exception as e:
                return {
                        "message": "Something went wrong!",
                        "error": str(e),
                        "data": None
                }, 500
    class Titanic(Resource):
        def post(self):
            passenger_data = request.get_json()

            return jsonify(response)
            
    # building RESTapi endpoint
    api.add_resource(_CRUD, '/')
    api.add_resource(_DesignCRUD, '/design')
    api.add_resource(_SearchCRUD, '/search')
    api.add_resource(Titanic, '/titanic')
    api.add_resource(Images,'/images')
    api.add_resource(_Security, '/authenticate')
    