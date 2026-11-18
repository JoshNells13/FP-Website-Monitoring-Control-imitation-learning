"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Line } from "@react-three/drei";
import * as THREE from "three";
import { STLLoader } from "three-stdlib";
import { API_BASE, CameraPOV, TrajectoryFrame } from "@/lib/api";

export interface BodyMeshSpec {
  index: number;
  body_name: string;
  parent: string | null;
  mesh: string;
  material: string;
}

const DEFAULT_BODY_SPECS: BodyMeshSpec[] = [
  { index: 0, body_name: "Trunk", parent: null, mesh: "Trunk.STL", material: "silver" },
  { index: 1, body_name: "Head_1", parent: "Trunk", mesh: "Head_1.STL", material: "silver" },
  { index: 2, body_name: "Head_2", parent: "Head_1", mesh: "Head_2.STL", material: "silver" },
  { index: 3, body_name: "Left_Arm_1", parent: "Trunk", mesh: "Left_Arm_1.STL", material: "silver" },
  { index: 4, body_name: "Left_Arm_2", parent: "Left_Arm_1", mesh: "Left_Arm_2.STL", material: "silver" },
  { index: 5, body_name: "Left_Arm_3", parent: "Left_Arm_2", mesh: "Left_Arm_3.STL", material: "silver" },
  { index: 6, body_name: "left_hand_link", parent: "Left_Arm_3", mesh: "Left_Arm_4.STL", material: "silver" },
  { index: 7, body_name: "Right_Arm_1", parent: "Trunk", mesh: "Right_Arm_1.STL", material: "silver" },
  { index: 8, body_name: "Right_Arm_2", parent: "Right_Arm_1", mesh: "Right_Arm_2.STL", material: "silver" },
  { index: 9, body_name: "Right_Arm_3", parent: "Right_Arm_2", mesh: "Right_Arm_3.STL", material: "silver" },
  { index: 10, body_name: "right_hand_link", parent: "Right_Arm_3", mesh: "Right_Arm_4.STL", material: "silver" },
  { index: 11, body_name: "Left_Hip_Pitch", parent: "Trunk", mesh: "Left_Hip_Pitch.STL", material: "silver" },
  { index: 12, body_name: "Left_Hip_Roll", parent: "Left_Hip_Pitch", mesh: "Left_Hip_Roll.STL", material: "silver" },
  { index: 13, body_name: "Left_Hip_Yaw", parent: "Left_Hip_Roll", mesh: "Left_Hip_Yaw.STL", material: "silver" },
  { index: 14, body_name: "Left_Shank", parent: "Left_Hip_Yaw", mesh: "Left_Shank.STL", material: "silver" },
  { index: 15, body_name: "Left_Ankle_Cross", parent: "Left_Shank", mesh: "Left_Ankle_Cross.STL", material: "silver" },
  { index: 16, body_name: "left_foot_link", parent: "Left_Ankle_Cross", mesh: "Left_Foot.STL", material: "dark" },
  { index: 17, body_name: "Right_Hip_Pitch", parent: "Trunk", mesh: "Right_Hip_Pitch.STL", material: "silver" },
  { index: 18, body_name: "Right_Hip_Roll", parent: "Right_Hip_Pitch", mesh: "Right_Hip_Roll.STL", material: "silver" },
  { index: 19, body_name: "Right_Hip_Yaw", parent: "Right_Hip_Roll", mesh: "Right_Hip_Yaw.STL", material: "silver" },
  { index: 20, body_name: "Right_Shank", parent: "Right_Hip_Yaw", mesh: "Right_Shank.STL", material: "silver" },
  { index: 21, body_name: "Right_Ankle_Cross", parent: "Right_Shank", mesh: "Right_Ankle_Cross.STL", material: "silver" },
  { index: 22, body_name: "right_foot_link", parent: "Right_Ankle_Cross", mesh: "Right_Foot.STL", material: "dark" },
];

const PARENT_INDICES: number[] = [
  -1, 0, 1, 0, 3, 4, 5, 0, 7, 8, 9, 0, 11, 12, 13, 14, 15, 0, 17, 18, 19, 20, 21,
];

const geometryCache: Record<string, THREE.BufferGeometry> = {};

function useBoosterMeshes(specs: BodyMeshSpec[]) {
  const [geometries, setGeometries] = useState<Record<string, THREE.BufferGeometry>>({ ...geometryCache });

  useEffect(() => {
    let active = true;
    const loader = new STLLoader();
    const uniqueMeshes = Array.from(new Set(specs.map((s) => s.mesh)));

    uniqueMeshes.forEach((meshName) => {
      if (geometryCache[meshName]) return;
      loader.load(
        `${API_BASE}/api/simulation/meshes/${meshName}`,
        (geom) => {
          geom.computeVertexNormals();
          geometryCache[meshName] = geom;
          if (active) {
            setGeometries((prev) => ({ ...prev, [meshName]: geom }));
          }
        },
        undefined,
        () => {
          // Fallback geometry if network hiccups
        }
      );
    });

    return () => {
      active = false;
    };
  }, [specs]);

  return geometries;
}

interface HumanoidModelProps {
  specs: BodyMeshSpec[];
  geometries: Record<string, THREE.BufferGeometry>;
  bodyPos: number[][];
  bodyQuat: number[][];
  bodyErrorMm?: number[];
  yOffset: number;
  variant: "policy" | "reference";
  showHeatmap: boolean;
  showSkeleton: boolean;
  hideHeadForFpv?: boolean;
  selectedBodyIndex?: number | null;
  onSelectBody?: (idx: number) => void;
}

function HumanoidModel({
  specs,
  geometries,
  bodyPos,
  bodyQuat,
  bodyErrorMm,
  yOffset,
  variant,
  showHeatmap,
  showSkeleton,
  hideHeadForFpv,
  selectedBodyIndex,
  onSelectBody,
}: HumanoidModelProps) {
  const isRef = variant === "reference";

  const skeletonSegments = useMemo(() => {
    if (!showSkeleton || !bodyPos || bodyPos.length < 23) return [];
    const segs: [THREE.Vector3, THREE.Vector3][] = [];
    for (let i = 1; i < 23; i++) {
      const pIdx = PARENT_INDICES[i];
      if (pIdx >= 0 && bodyPos[i] && bodyPos[pIdx]) {
        segs.push([
          new THREE.Vector3(bodyPos[pIdx][0], bodyPos[pIdx][1] + yOffset, bodyPos[pIdx][2]),
          new THREE.Vector3(bodyPos[i][0], bodyPos[i][1] + yOffset, bodyPos[i][2]),
        ]);
      }
    }
    return segs;
  }, [showSkeleton, bodyPos, yOffset]);

  return (
    <group>
      {specs.map((spec) => {
        const idx = spec.index;
        const pos = bodyPos?.[idx] || [0, 0, 0.5];
        const quatWxyz = bodyQuat?.[idx] || [1, 0, 0, 0];
        const geom = geometries[spec.mesh];

        if (hideHeadForFpv && (idx === 1 || idx === 2)) {
          return null;
        }

        // Color computation
        let color = isRef ? "#38bdf8" : spec.material === "dark" ? "#334155" : "#e2e8f0";
        let emissive = "#000000";
        let emissiveIntensity = 0;

        if (!isRef && showHeatmap && bodyErrorMm && bodyErrorMm[idx] !== undefined) {
          const err = bodyErrorMm[idx];
          if (err > 75) {
            color = "#f43f5e";
            emissive = "#be123c";
            emissiveIntensity = 0.35;
          } else if (err > 42) {
            color = "#fb923c";
            emissive = "#c2410c";
            emissiveIntensity = 0.2;
          } else {
            color = "#34d399";
          }
        }

        if (selectedBodyIndex === idx) {
          emissive = "#38bdf8";
          emissiveIntensity = 0.55;
        }

        const quaternion = new THREE.Quaternion(
          quatWxyz[1],
          quatWxyz[2],
          quatWxyz[3],
          quatWxyz[0]
        );

        return (
          <group
            key={spec.body_name}
            position={[pos[0], pos[1] + yOffset, pos[2]]}
            quaternion={quaternion}
            onClick={(e) => {
              e.stopPropagation();
              onSelectBody?.(idx);
            }}
          >
            {geom ? (
              <mesh geometry={geom} castShadow={!isRef} receiveShadow={!isRef}>
                <meshStandardMaterial
                  color={color}
                  emissive={emissive}
                  emissiveIntensity={emissiveIntensity}
                  metalness={isRef ? 0.15 : 0.45}
                  roughness={isRef ? 0.5 : 0.35}
                  transparent={isRef}
                  opacity={isRef ? 0.45 : 1.0}
                  wireframe={false}
                />
              </mesh>
            ) : (
              <mesh>
                <sphereGeometry args={[0.032, 12, 12]} />
                <meshStandardMaterial
                  color={color}
                  transparent={isRef}
                  opacity={isRef ? 0.45 : 1.0}
                />
              </mesh>
            )}
          </group>
        );
      })}

      {skeletonSegments.map((seg, i) => (
        <Line
          key={i}
          points={seg}
          color={isRef ? "#38bdf8" : "#facc15"}
          lineWidth={2}
        />
      ))}
    </group>
  );
}

interface CameraControllerProps {
  cameraPov: CameraPOV;
  frameData: TrajectoryFrame | null;
  yOffsetPolicy: number;
}

function CameraController({ cameraPov, frameData, yOffsetPolicy }: CameraControllerProps) {
  const { camera } = useThree();
  const orbitRef = useRef<any>(null);

  useEffect(() => {
    camera.up.set(0, 0, 1);
  }, [camera, cameraPov]);

  useFrame(() => {
    if (!frameData) return;
    camera.up.set(0, 0, 1);

    const trunkPos = frameData.sim_body_pos?.[0] || [0, 0, 0.53];
    const tx = trunkPos[0];
    const ty = trunkPos[1] + yOffsetPolicy;
    const tz = trunkPos[2];

    if (cameraPov === "free") {
      if (orbitRef.current) {
        orbitRef.current.target.lerp(new THREE.Vector3(tx, 0, tz), 0.12);
        orbitRef.current.update();
      }
      return;
    }

    if (cameraPov === "first_person") {
      const headPos = frameData.sim_body_pos?.[2] || [tx, ty, tz + 0.24];
      const headQuatWxyz = frameData.sim_body_quat?.[2] || [1, 0, 0, 0];
      const hq = new THREE.Quaternion(
        headQuatWxyz[1],
        headQuatWxyz[2],
        headQuatWxyz[3],
        headQuatWxyz[0]
      );

      // Eye offset on Head_2 (ZED stereo camera location at front of head)
      const eyeOffset = new THREE.Vector3(0.068, 0.0, 0.082).applyQuaternion(hq);
      const eyeWorld = new THREE.Vector3(
        headPos[0] + eyeOffset.x,
        headPos[1] + yOffsetPolicy + eyeOffset.y,
        headPos[2] + eyeOffset.z
      );

      // Forward look vector (+X forward, slightly pitched down -0.14 in Z)
      const forwardVec = new THREE.Vector3(1.2, 0.0, -0.16).applyQuaternion(hq);
      const upVec = new THREE.Vector3(0, 0, 1).applyQuaternion(hq);

      camera.position.copy(eyeWorld);
      camera.up.copy(upVec);
      camera.lookAt(
        eyeWorld.x + forwardVec.x,
        eyeWorld.y + forwardVec.y,
        eyeWorld.z + forwardVec.z
      );
      return;
    }

    const target = new THREE.Vector3(tx, yOffsetPolicy * 0.35, tz);
    let desiredPos = new THREE.Vector3(tx - 1.85, ty, tz + 0.35);

    if (cameraPov === "third_person") {
      desiredPos.set(tx - 1.85, ty, tz + 0.38);
    } else if (cameraPov === "front") {
      desiredPos.set(tx + 1.85, ty, tz + 0.25);
    } else if (cameraPov === "side") {
      desiredPos.set(tx, ty - 2.1, tz + 0.22);
    } else if (cameraPov === "top") {
      desiredPos.set(tx - 0.08, ty, tz + 2.45);
    }

    camera.position.lerp(desiredPos, 0.22);
    camera.lookAt(target);
  });

  if (cameraPov === "free") {
    return (
      <OrbitControls
        ref={orbitRef}
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={0.4}
        maxDistance={8.0}
      />
    );
  }
  return null;
}

function ZUpGroundGrid() {
  return (
    <group>
      {/* Main floor plane in XY (Z = 0) */}
      <mesh position={[0, 0, -0.002]} receiveShadow>
        <planeGeometry args={[16, 16]} />
        <meshStandardMaterial color="#090e17" roughness={0.85} metalness={0.1} />
      </mesh>
      {/* Grid rotated into XY plane */}
      <gridHelper
        args={[12, 24, "#1e3a8a", "#1e293b"]}
        rotation={[Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
      />
      {/* Origin axes indicator */}
      <axesHelper args={[0.35]} />
    </group>
  );
}

export interface RobotCanvas3DProps {
  frameData: TrajectoryFrame | null;
  cameraPov: CameraPOV;
  viewMode: "side_by_side" | "overlay" | "policy_only" | "reference_only";
  showHeatmap?: boolean;
  showSkeleton?: boolean;
  selectedBodyIndex?: number | null;
  onSelectBody?: (idx: number) => void;
  heightClass?: string;
}

export default function RobotCanvas3D({
  frameData,
  cameraPov,
  viewMode,
  showHeatmap = true,
  showSkeleton = false,
  selectedBodyIndex = null,
  onSelectBody,
  heightClass = "h-[540px]",
}: RobotCanvas3DProps) {
  const geometries = useBoosterMeshes(DEFAULT_BODY_SPECS);
  const loadedCount = Object.keys(geometries).length;

  const isSideBySide = viewMode === "side_by_side";
  const yOffsetRef = isSideBySide ? 0.52 : 0.0;
  const yOffsetPolicy = isSideBySide ? -0.52 : 0.0;

  return (
    <div className={`relative w-full ${heightClass} bg-[#060a11] rounded-xl overflow-hidden border border-slate-800/90 shadow-inner`}>
      <Canvas
        shadows
        camera={{ position: [-1.85, -0.4, 0.9], fov: 50, near: 0.04, far: 40 }}
      >
        <color attach="background" args={["#060a11"]} />
        <fog attach="fog" args={["#060a11", 6, 18]} />

        <ambientLight intensity={0.75} />
        <directionalLight
          position={[3, -3, 5]}
          intensity={1.35}
          castShadow
          shadow-mapSize-width={1024}
          shadow-mapSize-height={1024}
        />
        <directionalLight position={[-3, 3, 3]} intensity={0.65} color="#93c5fd" />
        <pointLight position={[0, 0, 2.5]} intensity={0.5} color="#38bdf8" />

        <ZUpGroundGrid />

        {frameData && (viewMode === "side_by_side" || viewMode === "overlay" || viewMode === "reference_only") && (
          <HumanoidModel
            specs={DEFAULT_BODY_SPECS}
            geometries={geometries}
            bodyPos={frameData.ref_body_pos}
            bodyQuat={frameData.ref_body_quat}
            yOffset={yOffsetRef}
            variant="reference"
            showHeatmap={false}
            showSkeleton={showSkeleton}
            hideHeadForFpv={false}
            selectedBodyIndex={selectedBodyIndex}
            onSelectBody={onSelectBody}
          />
        )}

        {frameData && (viewMode === "side_by_side" || viewMode === "overlay" || viewMode === "policy_only") && (
          <HumanoidModel
            specs={DEFAULT_BODY_SPECS}
            geometries={geometries}
            bodyPos={frameData.sim_body_pos}
            bodyQuat={frameData.sim_body_quat}
            bodyErrorMm={frameData.body_error_mm}
            yOffset={yOffsetPolicy}
            variant="policy"
            showHeatmap={showHeatmap}
            showSkeleton={showSkeleton}
            hideHeadForFpv={cameraPov === "first_person"}
            selectedBodyIndex={selectedBodyIndex}
            onSelectBody={onSelectBody}
          />
        )}

        <CameraController
          cameraPov={cameraPov}
          frameData={frameData}
          yOffsetPolicy={viewMode === "reference_only" ? yOffsetRef : yOffsetPolicy}
        />
      </Canvas>

      {/* Top-Left HUD Badge */}
      <div className="pointer-events-none absolute top-3 left-3 flex flex-col gap-1.5">
        <div className="flex items-center gap-2 bg-slate-950/85 backdrop-blur-md border border-slate-800 px-3 py-1.5 rounded-lg text-xs">
          <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="font-semibold text-slate-100">Booster K1 (22-DoF)</span>
          <span className="text-slate-400">|</span>
          <span className="font-mono text-sky-400 uppercase">{cameraPov.replace("_", " ")} POV</span>
          <span className="text-slate-400">|</span>
          <span className="font-mono text-slate-300">
            STL Meshes: {loadedCount}/23
          </span>
        </div>

        {(viewMode === "side_by_side" || viewMode === "overlay") && (
          <div className="flex items-center gap-3 bg-slate-950/80 backdrop-blur-md border border-slate-800/80 px-3 py-1 rounded-lg text-[11px]">
            <span className="flex items-center gap-1.5 text-sky-400">
              <span className="h-2.5 w-2.5 rounded-sm bg-sky-400/60 border border-sky-400" />
              GMR Reference (Kinematic)
            </span>
            <span className="flex items-center gap-1.5 text-slate-200">
              <span className="h-2.5 w-2.5 rounded-sm bg-emerald-400" />
              BeyondMimic Policy (Closed-Loop)
            </span>
          </div>
        )}
      </div>

      {/* First-Person Robotic Visor Reticle Overlay */}
      {cameraPov === "first_person" && frameData && (
        <div className="pointer-events-none absolute inset-0 border-2 border-sky-500/30 rounded-xl flex flex-col justify-between p-4">
          <div className="flex justify-between items-start">
            <div className="bg-slate-950/80 border border-sky-500/40 px-2.5 py-1 rounded text-[11px] font-mono text-sky-300">
              HEAD_2 ZED-FPV SENSOR // POS: [
              {frameData.head_camera_pose.pos.map((v) => v.toFixed(2)).join(", ")}]m
            </div>
            <div className="bg-slate-950/80 border border-sky-500/40 px-2.5 py-1 rounded text-[11px] font-mono text-emerald-300">
              ANCHOR TILT ERR: {frameData.frame_anchor_ori_err_deg.toFixed(2)}°
            </div>
          </div>
          <div className="self-center my-auto flex items-center justify-center opacity-45">
            <div className="w-24 h-24 rounded-full border border-sky-400/60 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-sky-400" />
            </div>
          </div>
          <div className="text-center text-[11px] font-mono text-sky-400/80">
            FIRST-PERSON HEAD CAMERA POSE DERIVED FROM HEAD_2 LINK STATE
          </div>
        </div>
      )}

      {/* Bottom-Right Live Frame Telemetry Overlay */}
      {frameData && (
        <div className="pointer-events-none absolute bottom-3 right-3 bg-slate-950/85 backdrop-blur-md border border-slate-800 px-3.5 py-2 rounded-lg text-xs font-mono flex items-center gap-4">
          <div>
            <span className="text-slate-400">FRAME: </span>
            <span className="text-slate-100 font-semibold">{frameData.frame + 1}/149</span>
          </div>
          <div>
            <span className="text-slate-400">TIME: </span>
            <span className="text-sky-400 font-semibold">{frameData.time_sec.toFixed(2)}s</span>
          </div>
          <div>
            <span className="text-slate-400">MAE: </span>
            <span className="text-emerald-400 font-semibold">{frameData.frame_mae_rad.toFixed(3)} rad</span>
          </div>
          <div>
            <span className="text-slate-400">R-MPKPE: </span>
            <span className="text-amber-400 font-semibold">{frameData.frame_rmpkpe_mm.toFixed(1)} mm</span>
          </div>
        </div>
      )}
    </div>
  );
}
