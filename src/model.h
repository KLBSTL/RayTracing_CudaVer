//
// Created by 86189 on 26-6-8.
//

#ifndef MODEL_H
#define MODEL_H

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "mesh.h"

class Model {
public:
    explicit Model(std::string path) {
        load_model(std::move(path));
    }

    bool loaded() const {
        return loaded_;
    }

    const aabb& bounding_box() const {
        return bbox;
    }

    void append_to_scene(
        std::vector<primitive>& prims,
        std::vector<triangleData>& triangles,
        std::vector<Material>& mats,
        const Material& mat,
        vec3 target,
        float scale,
        float theta
    ) const {
        vec3 min_p( infinity,  infinity,  infinity);
        vec3 max_p(-infinity, -infinity, -infinity);

        vec3 center(
            0.5f * (bbox.x.min + bbox.x.max),
            0.5f * (bbox.y.min + bbox.y.max),
            0.5f * (bbox.z.min + bbox.z.max)
        );

        for (const mesh& parsed_mesh : meshes) {
            parsed_mesh.append_to_scene(prims, triangles, mats, mat,center,scale,theta,target);
        }
    }

private:
    void load_model(std::string path) {
        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(
            path,
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices |
            aiProcess_ImproveCacheLocality |
            aiProcess_SortByPType
        );

        if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) {
            std::cerr << "ERROR::ASSIMP:: " << importer.GetErrorString() << "\n";
            return;
        }

        const size_t last_slash = path.find_last_of("/\\");
        directory = last_slash == std::string::npos ? "." : path.substr(0, last_slash);
        process_node(scene->mRootNode, scene);
        loaded_ = !meshes.empty();
    }

    void process_node(aiNode* node, const aiScene* scene) {
        for (unsigned int i = 0; i < node->mNumMeshes; ++i) {
            aiMesh* assimp_mesh = scene->mMeshes[node->mMeshes[i]];
            mesh parsed_mesh = process_mesh(assimp_mesh);
            bbox = aabb(bbox, parsed_mesh.bounding_box());
            meshes.push_back(std::move(parsed_mesh));
        }

        for (unsigned int i = 0; i < node->mNumChildren; ++i) {
            process_node(node->mChildren[i], scene);
        }
    }

    mesh process_mesh(const aiMesh* assimp_mesh) const {
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;

        vertices.reserve(assimp_mesh->mNumVertices);
        for (unsigned int i = 0; i < assimp_mesh->mNumVertices; ++i) {
            Vertex vertex{};

            vertex.Position = vec3(
                assimp_mesh->mVertices[i].x,
                assimp_mesh->mVertices[i].y,
                assimp_mesh->mVertices[i].z
            );

            if (assimp_mesh->HasNormals()) {
                vertex.Normal = vec3(
                    assimp_mesh->mNormals[i].x,
                    assimp_mesh->mNormals[i].y,
                    assimp_mesh->mNormals[i].z
                );
            }

            if (assimp_mesh->mTextureCoords[0]) {
                vertex.TexCoords = vec2(
                    assimp_mesh->mTextureCoords[0][i].x,
                    assimp_mesh->mTextureCoords[0][i].y
                );
            }

            if (assimp_mesh->HasTangentsAndBitangents()) {
                vertex.Tangent = vec3(
                    assimp_mesh->mTangents[i].x,
                    assimp_mesh->mTangents[i].y,
                    assimp_mesh->mTangents[i].z
                );
                vertex.Bitangent = vec3(
                    assimp_mesh->mBitangents[i].x,
                    assimp_mesh->mBitangents[i].y,
                    assimp_mesh->mBitangents[i].z
                );
            }

            vertices.push_back(vertex);
        }

        for (unsigned int i = 0; i < assimp_mesh->mNumFaces; ++i) {
            const aiFace& face = assimp_mesh->mFaces[i];
            if (face.mNumIndices != 3) {
                continue;
            }

            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }

        return mesh(std::move(vertices), std::move(indices));
    }

    std::vector<mesh> meshes;
    std::string directory;
    aabb bbox = aabb::empty();
    bool loaded_ = false;
};

#endif //MODEL_H
