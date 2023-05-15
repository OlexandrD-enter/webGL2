"use strict";


// Vertex shader
const vertexShaderSource = `
attribute vec3 vertex;
attribute vec2 texCoord;
uniform mat4 ModelViewProjectionMatrix;
varying vec2 v_texcoord;

attribute vec3 a_normal;
varying vec3 v_normal;

uniform mat4 u_worldViewProjection;
uniform mat4 u_world;

void main() {
    gl_Position = ModelViewProjectionMatrix * vec4(vertex,1.0);
    v_texcoord = texCoord;
	v_normal = mat3(u_world) * a_normal;
}`;


// Fragment shader
const fragmentShaderSource = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
   precision highp float;
#else
   precision mediump float;
#endif

uniform sampler2D u_texture;
uniform float fColorCoef;

varying vec2 v_texcoord;
uniform vec4 color;

varying vec3 v_normal;
uniform vec3 u_reverseLightDirection;
uniform vec4 u_color;

void main() {
	vec3 normal = normalize(v_normal);
	float light = dot(normal, u_reverseLightDirection);
	
    gl_FragColor = color*fColorCoef + texture2D(u_texture, v_texcoord)*(1.0-fColorCoef);
	
	gl_FragColor.rgb *= light;
}`;

let gl; // The webgl context.

let iAttribVertex; // Location of the attribute variable in the shader program.
let iAttribTexture; // Location of the attribute variable in the shader program.

let iColor; // Location of the uniform specifying a color for the primitive.
let iColorCoef; // Location of the uniform specifying a color for the primitive.
let iModelViewProjectionMatrix; // Location of the uniform matrix representing the combined transformation.
let iTextureMappingUnit;

let iVertexBuffer; // Buffer to hold the values.
let iTexBuffer; // Buffer to hold the values.

let spaceball; // A SimpleRotator object that lets the user rotate the view by mouse.

let reverseLightDirectionLocation;
let colorLocation;
let normalLocation;
let normalBuffer;

let worldViewProjectionLocation;
let worldLocation;

let scale = 1.0;
let convergence = 30;
let eyeSeparation = 0.1;
let FOV = Math.PI / 8;
let nearClippingDistance = 7;

let AnaglyphCamera;

function degToRad(d) {
  return (d * Math.PI) / 180;
}

function drawPrimitive(primitiveType, color, vertices, normals, texCoords) {
  gl.uniform4fv(iColor, color);
  gl.uniform1f(iColorCoef, 0.0);

  gl.uniform3fv(reverseLightDirectionLocation, m4.normalize([0, 0, 1]));

  gl.enableVertexAttribArray(iAttribVertex);
  gl.bindBuffer(gl.ARRAY_BUFFER, iVertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STREAM_DRAW);
  gl.vertexAttribPointer(iAttribVertex, 3, gl.FLOAT, false, 0, 0);

  gl.enableVertexAttribArray(normalLocation);
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);
  gl.vertexAttribPointer(normalLocation, 3, gl.FLOAT, false, 0, 0);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(normals), gl.STATIC_DRAW);

  if (texCoords) {
    gl.enableVertexAttribArray(iAttribTexture);
    gl.bindBuffer(gl.ARRAY_BUFFER, iTexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(texCoords), gl.STATIC_DRAW);
    gl.vertexAttribPointer(iAttribTexture, 2, gl.FLOAT, false, 0, 0);
  } else {
    gl.disableVertexAttribArray(iAttribTexture);
    gl.vertexAttrib2f(iAttribTexture, 0.0, 0.0);
    gl.uniform1f(iColorCoef, 1.0);
  }

  gl.drawArrays(primitiveType, 0, vertices.length / 3);
}

// Constructor function
function StereoCamera(
  Convergence,
  EyeSeparation,
  AspectRatio,
  FOV,
  NearClippingDistance,
  FarClippingDistance
) {
  this.mConvergence = Convergence;
  this.mEyeSeparation = EyeSeparation;
  this.mAspectRatio = AspectRatio;
  this.mFOV = FOV;
  this.mNearClippingDistance = NearClippingDistance;
  this.mFarClippingDistance = FarClippingDistance;

  this.mLeftProjectionMatrix = null;
  this.mRightProjectionMatrix = null;

  this.mLeftModelViewMatrix = null;
  this.mRightModelViewMatrix = null;

  this.ApplyLeftFrustum = function () {
    let top, bottom, left, right;
    top = this.mNearClippingDistance * Math.tan(this.mFOV / 2);
    bottom = -top;

    let a = this.mAspectRatio * Math.tan(this.mFOV / 2) * this.mConvergence;
    let b = a - this.mEyeSeparation / 2;
    let c = a + this.mEyeSeparation / 2;

    left = (-b * this.mNearClippingDistance) / this.mConvergence;
    right = (c * this.mNearClippingDistance) / this.mConvergence;

    // Set the Projection Matrix
    this.mLeftProjectionMatrix = m4.frustum(
      left,
      right,
      bottom,
      top,
      this.mNearClippingDistance,
      this.mFarClippingDistance
    );

    // Displace the world to right
    this.mLeftModelViewMatrix = m4.translation(
      this.mEyeSeparation / 2,
      0.0,
      0.0
    );
  };

  this.ApplyRightFrustum = function () {
    let top, bottom, left, right;
    top = this.mNearClippingDistance * Math.tan(this.mFOV / 2);
    bottom = -top;

    let a = this.mAspectRatio * Math.tan(this.mFOV / 2) * this.mConvergence;
    let b = a - this.mEyeSeparation / 2;
    let c = a + this.mEyeSeparation / 2;

    left = (-c * this.mNearClippingDistance) / this.mConvergence;
    right = (b * this.mNearClippingDistance) / this.mConvergence;

    // Set the Projection Matrix
    this.mRightProjectionMatrix = m4.frustum(
      left,
      right,
      bottom,
      top,
      this.mNearClippingDistance,
      this.mFarClippingDistance
    );

    // Displace the world to left
    this.mRightModelViewMatrix = m4.translation(
      -this.mEyeSeparation / 2,
      0.0,
      0.0
    );
  };
}

let X = (u, v) =>
  (R + a * Math.cos(u / 2)) * Math.cos(u / 3) +
  a * Math.cos(u / 3) * Math.cos(v - Math.PI);
let Y = (u, v) =>
  (R + a * Math.cos(u / 2)) * Math.sin(u / 3) +
  a * Math.sin(u / 3) * Math.cos(v - Math.PI);
let Z = (u, v) => a + Math.sin(u / 2) + a * Math.sin(v - Math.PI);

let diapazonUFrom = 0;
let diapazonUTo = 12 * Math.PI;
let diapazonVFrom = 0;
let diapazonVTo = 2 * Math.PI;
let step = 0.8;
let a = 0.25;
let R = 0.8;

function deg2rad(angle) {
  return (angle * Math.PI) / 180;
}

const calcDerU = (u, v, dU) => [
  (X(u + dU, v) - X(u, v)) / deg2rad(dU),
  (Y(u + dU, v) - Y(u, v)) / deg2rad(dU),
  (Z(u + dU, v) - Z(u, v)) / deg2rad(dU),
];

const calcDerV = (u, v, dV) => [
  (X(u, v + dV) - X(u, v)) / deg2rad(dV),
  (Y(u, v + dV) - Y(u, v)) / deg2rad(dV),
  (Z(u, v + dV) - Z(u, v)) / deg2rad(dV),
];

function DrawSurface() {
  gl.enableVertexAttribArray(iAttribTexture);
  gl.bindBuffer(gl.ARRAY_BUFFER, iTexBuffer);
  gl.bufferData(
    gl.ARRAY_BUFFER,
    new Float32Array([
      [0, 0, 0],
      [0, 1, 0],
      [0, 1, 1],
      [0, 0, 1],
    ]),
    gl.STREAM_DRAW
  );
  gl.vertexAttribPointer(iAttribTexture, 2, gl.FLOAT, false, 0, 0);

  let vertexList = [];
  let normalsList = [];

  let deltaU = 0.001;
  let deltaV = 0.001;

  for (let u = diapazonUFrom; u <= diapazonUTo; u += step) {
    for (let v = diapazonVFrom; v <= diapazonVTo; v += step) {
      const u0 = u;
      const v0 = v;
      const u1 = u + step;
      const v1 = v + step;

      let x0 = X(u0, v0);
      let y0 = Y(u0, v0);
      let z0 = Z(u0, v0);

      let xR = X(u1, v0);
      let yR = Y(u1, v0);
      let zR = Z(u1, v0);

      vertexList.push(x0, z0, y0);
      vertexList.push(xR, zR, yR);

      normalsList.push(
        ...m4.cross(calcDerU(u0, v0, deltaU), calcDerV(u0, v0, deltaV))
      );
      normalsList.push(
        ...m4.cross(calcDerU(u1, v0, deltaU), calcDerV(u1, v0, deltaV))
      );
    }
    drawPrimitive(gl.TRIANGLE_STRIP, [0.5, 1, 0.5, 1], vertexList, normalsList); // цвет
  }
}

function rotationX(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [1, 0, 0, 0, 0, c, -s, 0, 0, s, c, 0, 0, 0, 0, 1];
}

function rotationY(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c, 0, s, 0, 0, 1, 0, 0, -s, 0, c, 0, 0, 0, 0, 1];
}

function rotationZ(angle) {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return [c, -s, 0, 0, s, c, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}

let rotationMatrix = m4.identity();

/*function requestDeviceOrientation() {
  if(typeof DeviceOrientationEvent !== 'undefined' &&
    typeof DeviceOrientationEvent.requestPermission === 'function') {
      DeviceOrientationEvent.requestPermission()
          .then(response => {
              console.log(response);
              if (response === 'granted') {
                  console.log('Permission granted');
                  window.addEventListener('deviceorientation', e => {...}, true); 
              }
          }).catch((err => {
          console.log('Err', err);
      }));
  } else
      console.log('not iOS');        
}*/

/*if (window.DeviceOrientationEvent) {
  if (typeof DeviceMotionEvent.requestPermission === 'function') {
    DeviceMotionEvent.requestPermission()
      .then(permissionState => {
        if (permissionState === 'granted') {
          window.addEventListener("deviceorientation", e => handleDeviceOrientation(e) , true);
        } else {
          console.log("DeviceOrientationEvent permission not granted");
        }
      })
      .catch(console.error);
  } else {
    // For browsers that don't support requestPermission()
    window.addEventListener("deviceorientation", e => handleDeviceOrientation(e), true);
  }
} else {
  console.log("DeviceOrientationEvent is not supported");
}*/

/*if ('Gyroscope' in window) {
  const sensor = new Gyroscope();
  sensor.addEventListener('reading', handleOrientation);
  sensor.start();
}

function handleOrientation() {
  const alpha = sensor.x; // Z-axis rotation
  const beta = sensor.y; // X-axis rotation
  const gamma = sensor.z; // Y-axis rotation

  rotationMatrix = m4.rotationZ(alpha) * m4.rotationX(beta) * m4.rotationY(gamma);
}*/

if ('Gyroscope' in window) {
  const sensor = new Gyroscope();
  sensor.addEventListener('reading', e => onSensorChanged(e));
  sensor.start();
}


const NS2S = 1.0 / 1000000000.0;
const deltaRotationVector = [0.0, 0.0, 0.0, 0.0];
let timestamp = 0.0;

function onSensorChanged(event) {
  // This timestep's delta rotation to be multiplied by the current rotation
  // after computing it from the gyro sample data.
  if (timestamp !== 0) {
    const dT = (event.timestamp - timestamp) * NS2S;
    // Axis of the rotation sample, not normalized yet.
    let axisX = event.values[0];
    let axisY = event.values[1];
    let axisZ = event.values[2];

    // Calculate the angular speed of the sample
    const omegaMagnitude = Math.sqrt(axisX * axisX + axisY * axisY + axisZ * axisZ);

    // Normalize the rotation vector if it's big enough to get the axis
    // (that is, EPSILON should represent your maximum allowable margin of error)
    const EPSILON = 0.000001;
    if (omegaMagnitude > EPSILON) {
      axisX /= omegaMagnitude;
      axisY /= omegaMagnitude;
      axisZ /= omegaMagnitude;
    }

    // Integrate around this axis with the angular speed by the timestep
    // in order to get a delta rotation from this sample over the timestep
    // We will convert this axis-angle representation of the delta rotation
    // into a quaternion before turning it into the rotation matrix.
    const thetaOverTwo = omegaMagnitude * dT / 2.0;
    const sinThetaOverTwo = Math.sin(thetaOverTwo);
    const cosThetaOverTwo = Math.cos(thetaOverTwo);
    deltaRotationVector[0] = sinThetaOverTwo * axisX;
    deltaRotationVector[1] = sinThetaOverTwo * axisY;
    deltaRotationVector[2] = sinThetaOverTwo * axisZ;
    deltaRotationVector[3] = cosThetaOverTwo;
  }
  timestamp = event.timestamp;
  const deltaRotationMatrix = new Float32Array(9);
  rotationMatrix = getRotationMatrixFromVector(deltaRotationMatrix, deltaRotationVector);
}

function getRotationMatrixFromVector(R, rotationVector) {
  let q0;
  const q1 = rotationVector[0];
  const q2 = rotationVector[1];
  const q3 = rotationVector[2];

  if (rotationVector.length >= 4) {
    q0 = rotationVector[3];
  } else {
    q0 = 1 - q1 * q1 - q2 * q2 - q3 * q3;
    q0 = (q0 > 0) ? Math.sqrt(q0) : 0;
  }

  const sq_q1 = 2 * q1 * q1;
  const sq_q2 = 2 * q2 * q2;
  const sq_q3 = 2 * q3 * q3;
  const q1_q2 = 2 * q1 * q2;
  const q3_q0 = 2 * q3 * q0;
  const q1_q3 = 2 * q1 * q3;
  const q2_q0 = 2 * q2 * q0;
  const q2_q3 = 2 * q2 * q3;
  const q1_q0 = 2 * q1 * q0;

  if (R.length === 9) {
    R[0] = 1 - sq_q2 - sq_q3;
    R[1] = q1_q2 - q3_q0;
    R[2] = q1_q3 + q2_q0;
    R[3] = q1_q2 + q3_q0;
    R[4] = 1 - sq_q1 - sq_q3;
    R[5] = q2_q3 - q1_q0;
    R[6] = q1_q3 - q2_q0;
    R[7] = q2_q3 + q1_q0;
    R[8] = 1 - sq_q1 - sq_q2;
  } else if (R.length === 16) {
    R[0] = 1 - sq_q2 - sq_q3;
    R[1] = q1_q2 - q3_q0;
    R[2] = q1_q3 + q2_q0;
    R[3] = 0.0;
    R[4] = q1_q2 + q3_q0;
    R[5] = 1 - sq_q1 - sq_q3;
    R[6] = q2_q3 - q1_q0;
    R[7] = 0.0;
    R[8] = q1_q3 - q2_q0;
    R[9] = q2_q3 + q1_q0;
    R[10] = 1 - sq_q1 - sq_q2;
    R[11] = 0.0;
    R[12] = R[13] = R[14] = 0.0;
    R[15] = 1.0;
  }
  return R;
}



/*function handleDeviceOrientation(event){
  if (event.alpha === null || event.beta === null || event.gamma === null) {
    console.log("Device orientation data is not available");
    return;
  }

  const alpha = (event.alpha * Math.PI) / 180;
  const beta = (event.beta * Math.PI) / 180;
  const gamma = (event.gamma * Math.PI) / 180;
  
  const Rx = rotationX(beta);
  const Ry = rotationY(gamma);
  const Rz = rotationZ(alpha);
  rotationMatrix = m4.multiply(Rz, m4.multiply(Rx, Ry));
  draw();
}*/


/* Draws a colored cube, along with a set of coordinate axes.
 * (Note that the use of the above drawPrimitive function is not an efficient
 * way to draw with WebGL.  Here, the geometry is so simple that it doesn't matter.)
 */
function draw() {
  AnaglyphCamera = new StereoCamera(
    convergence,
    eyeSeparation,
    1,
    FOV,
    nearClippingDistance,
    12
  );
  AnaglyphCamera.ApplyLeftFrustum();
  AnaglyphCamera.ApplyRightFrustum();

  gl.clearColor(0.9, 0.9, 0.9, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  /* Set the values of the projection transformation */
  let projection = AnaglyphCamera.mLeftProjectionMatrix;
  let scaleM = m4.scaling(scale, scale, scale);

  /* Get the view matrix from the SimpleRotator object.*/
  let modelView = rotationMatrix;

  let rotateToPointZero = m4.axisRotation([-0.5, 0.2, 0.3], 1.4);
  let translateToPointZero = m4.translation(0, 0, -10);

  let matAccum0 = m4.multiply(scaleM, modelView);
  let matAccum1 = m4.multiply(rotateToPointZero, matAccum0);
  let matAccum2 = m4.multiply(translateToPointZero, matAccum1);
  let matAccum3 = m4.multiply(AnaglyphCamera.mLeftModelViewMatrix, matAccum2);

  /* Multiply the projection matrix times the modelview matrix to give the
       combined transformation matrix, and send that to the shader program. */
  let modelViewProjection = m4.multiply(projection, matAccum3);

  gl.uniformMatrix4fv(iModelViewProjectionMatrix, false, modelViewProjection);
  gl.uniform1i(iTextureMappingUnit, 0);

  //gl.uniformMatrix4fv(worldViewProjectionLocation, false, matAccum0);
  gl.uniformMatrix4fv(worldLocation, false, matAccum1);

  gl.colorMask(true, false, false, false);
  DrawSurface();

  gl.clear(gl.DEPTH_BUFFER_BIT);

  projection = AnaglyphCamera.mRightProjectionMatrix;
  matAccum3 = m4.multiply(AnaglyphCamera.mRightModelViewMatrix, matAccum2);
  modelViewProjection = m4.multiply(projection, matAccum3);

  gl.uniformMatrix4fv(iModelViewProjectionMatrix, false, modelViewProjection);

  gl.colorMask(false, true, true, false);
  DrawSurface();
  gl.colorMask(true, true, true, true);
}

/* Initialize the WebGL context. Called from init() */
function initWebGL() {
  const program = createProgram(gl, vertexShaderSource, fragmentShaderSource);
  gl.useProgram(program);

  iAttribVertex = gl.getAttribLocation(program, "vertex");
  iAttribTexture = gl.getAttribLocation(program, "texCoord");
  normalLocation = gl.getAttribLocation(program, "a_normal");

  iModelViewProjectionMatrix = gl.getUniformLocation(
    program,
    "ModelViewProjectionMatrix"
  );
  iColor = gl.getUniformLocation(program, "color");
  iColorCoef = gl.getUniformLocation(program, "fColorCoef");
  iTextureMappingUnit = gl.getUniformLocation(program, "u_texture");
  colorLocation = gl.getUniformLocation(program, "u_color");
  reverseLightDirectionLocation = gl.getUniformLocation(
    program,
    "u_reverseLightDirection"
  );

  worldLocation = gl.getUniformLocation(program, "u_world");

  iVertexBuffer = gl.createBuffer();
  iTexBuffer = gl.createBuffer();

  // Создаём буфер для нормалей
  normalBuffer = gl.createBuffer();
  // ARRAY_BUFFER = normalBuffer
  gl.bindBuffer(gl.ARRAY_BUFFER, normalBuffer);

  webglLessonsUI.setupSlider("#convergence", {
    value: convergence,
    slide: updateConvergence,
    min: 1,
    max: 100,
  });
  webglLessonsUI.setupSlider("#eyeSeparation", {
    value: eyeSeparation,
    slide: updateEyeSeparation,
    min: 0.01,
    max: 0.5,
    precision: 2,
    step: 0.01,
  });
  webglLessonsUI.setupSlider("#FOV", {
    value: FOV,
    slide: updateFOV,
    min: 0.01,
    max: 1,
    precision: 2,
    step: 0.01,
  });
  webglLessonsUI.setupSlider("#nearClippingDistance", {
    value: nearClippingDistance,
    slide: updateNearClippingDistance,
    min: 1,
    max: 20,
    precision: 2,
    step: 0.01,
  });

  gl.enable(gl.DEPTH_TEST);
}

function updateConvergence(event, ui) {
  convergence = ui.value;
  draw();
}

function updateEyeSeparation(event, ui) {
  eyeSeparation = ui.value;
  draw();
}

function updateFOV(event, ui) {
  FOV = ui.value;
  draw();
}

function updateNearClippingDistance(event, ui) {
  nearClippingDistance = ui.value;
  draw();
}

function createProgram(gl, vShader, fShader) {
  const vsh = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vsh, vShader);
  gl.compileShader(vsh);
  if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
    throw new Error("Error in vertex shader:  " + gl.getShaderInfoLog(vsh));
  }

  const fsh = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fsh, fShader);
  gl.compileShader(fsh);
  if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
    throw new Error("Error in fragment shader:  " + gl.getShaderInfoLog(fsh));
  }

  const program = gl.createProgram();
  gl.attachShader(program, vsh);
  gl.attachShader(program, fsh);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    throw new Error("Link error in program:  " + gl.getProgramInfoLog(program));
  }

  return program;
}

/**
 * initialization function that will be called when the page has loaded
 */
function init() {
  let canvas;
  try {
    canvas = document.getElementById("webglcanvas");
    gl = canvas.getContext("webgl");
    if (!gl) {
      throw "Browser does not support WebGL";
    }
  } catch (e) {
    document.getElementById("canvas-holder").innerHTML =
      "<p>Sorry, could not get a WebGL graphics context.</p>";
    return;
  }
  try {
    initWebGL(); // initialize the WebGL graphics context
  } catch (e) {
    document.getElementById("canvas-holder").innerHTML =
      "<p>Sorry, could not initialize the WebGL graphics context: " +
      e +
      "</p>";
    return;
  }

  spaceball = new TrackballRotator(canvas, draw, 0);

  draw();
}
