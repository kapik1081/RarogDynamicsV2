# VTOL dynamics structure

Arrows mean **uses or contains**, not execution order or inheritance. Dashed
arrows mark optional terrain contact. The four partial models inherit the
integration and state-management implementation from `VTOLModel`, but select
their own derivatives in `testing_dynamics.py`.

```mermaid
flowchart LR
    subgraph models["Full and partial dynamics models"]
        full["VTOLModel<br/>model.py + dynamics.py<br/>state_derivative: all 14 states active"]
        bicopter["BicopterFullModel<br/>testing.py + testing_dynamics.py<br/>bicopter_full_derivative: no aerodynamics"]
        translation["BicopterTranslationalModel<br/>testing.py + testing_dynamics.py<br/>bicopter_translational_derivative<br/>Fixed attitude; p = q = r = 0"]
        rotation["BicopterRotationModel<br/>testing.py + testing_dynamics.py<br/>bicopter_rotation_derivative<br/>Fixed position and body linear velocity"]
        glider["AerodynamicsOnlyModel<br/>testing.py + testing_dynamics.py<br/>aerodynamics_only_derivative<br/>Fixed nacelle tilts; engine commands ignored"]
    end

    subgraph motion["Rigid-body dynamics and kinematics"]
        coupled["Coupled translation and rotation"]
        linear["Translation<br/>translational_acceleration: force / mass - omega cross velocity<br/>body_to_ned: position rates<br/>gravity_acceleration: body-frame gravity"]
        angular["Rotation<br/>angular_acceleration: inertia and gyroscopic coupling<br/>euler_rates: attitude rates"]
        coupled --> linear
        coupled --> angular
    end

    subgraph propulsion["Propulsion components - propulsion.py"]
        engines["Complete engine dynamics"]
        thrust["Propeller thrust forces<br/>propeller_thrust + thrust_forces<br/>Independent left and right speeds and actual tilts"]
        moments["Engine moments"]
        offset["Thrust-offset moment<br/>thrust_moment"]
        reaction["Propeller reaction moment<br/>propeller_torque + reaction_moment"]
        tilt["Nacelle tilt actuators<br/>tilt_rate<br/>Independent first-order lag and rate limits"]
        engines --> thrust
        engines --> moments
        engines --> tilt
        moments --> offset
        moments --> reaction
        offset --> thrust
    end

    subgraph aerodynamics["Aerodynamic components - aerodynamics.py"]
        aero["Aerodynamic loads"]
        air["AirData / air_data<br/>Airspeed, alpha, beta, dynamic pressure"]
        force["aerodynamic_force<br/>Lift, drag, sideforce<br/>wind_to_body transform from kinematics.py"]
        moment["aerodynamic_moment<br/>Static, stability, rate-damping<br/>and control-surface terms"]
        coefficients["Coefficient evaluation - coefficients.py<br/>Scalar constants or CoefficientMap<br/>Bilinear alpha/beta interpolation"]
        stall["force_coefficients + stall_blend<br/>Optional lift/drag stall blending"]
        aero --> air
        aero --> force
        aero --> moment
        force --> stall
        stall --> coefficients
        moment --> coefficients
    end

    subgraph shared["Shared model components"]
        runtime["Integration and state management - model.py<br/>step / get_state / time<br/>SciPy solve_ivp with RK45"]
        inputs["State and ControlInputs - types.py<br/>14-state layout and held control inputs"]
        parameters["ModelParameters - parameters.py<br/>Mass, geometry, environment, propulsion<br/>Inertia, AerodynamicCoefficients, StallParameters"]
        terrain["Optional terrain contact - terrain.py + model.py<br/>project_to_terrain / normal_acceleration / contact_derivative<br/>Flat ground; impact and liftoff events"]
        runtime --> inputs
        runtime --> parameters
    end

    full --> coupled
    full --> engines
    full --> aero
    bicopter --> coupled
    bicopter --> engines
    translation --> linear
    translation --> thrust
    translation --> tilt
    rotation --> angular
    rotation --> moments
    rotation --> tilt
    glider --> coupled
    glider --> aero

    models --> runtime
    full -.-> terrain
    bicopter -.-> terrain
    translation -.-> terrain
    glider -.-> terrain

    classDef fullModel fill:#dbeafe,stroke:#2563eb,color:#172554
    classDef partialModel fill:#e0f2fe,stroke:#0284c7,color:#0c4a6e
    class full fullModel
    class bicopter,translation,rotation,glider partialModel
```

The motion functions are defined in `dynamics.py` and `kinematics.py`.
`VTOLModel` assembles force and moment components through `calculate_loads`
and `Loads`; the partial derivatives compose only their enabled physics.
Terrain contact is unavailable in `BicopterRotationModel`.
